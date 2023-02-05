"""
Base classes for physical media and transceivers, with these classes handling the
low-level details of connecting and disconnecting transceivers from media. Subclasses
of these base classes define the specifics of how transceivers connect to media and
how data is transmitted over the media.

:author: Alf O'Kenney
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

__all__ = ['Medium', 'Transceiver']

log = logging.getLogger(__name__)


class Medium(ABC):
    """A base class for all physical media.

    Just as in real life, a :class:`Medium` is passive in its interactions with a
    :class:`Transceiver`. That is, a :class:`Medium` instance does not initiate any
    connections or disconnections, nor does it request data. In PyNet, it exists merely as
    a record-keeper of the :class:`Transceiver` instances that are connected to it and as
    a means of transporting data between those :class:`Transceiver` instances in a manner
    that simulates the physical medium.

    Subclasses of :class:`Medium` define their own connection and disconnection logic by
    overriding the :meth:`_subclass_connect` and :meth:`_subclass_disconnect` methods.
    Examples of such logic may include registering :class:`Transceiver` instances as a
    possible recipient of data or confirming that there is enough space on the physical
    medium to accommodate a new connection (e.g., a coaxial cable to which 802.3 Ethernet
    transceivers connect via vampire taps).

    NOTE: Instances of :class:`Medium` subclasses must be interacted with via methods of
    a :class:`Transceiver` instance. That is, if a :class:`Medium` instance is to be
    connected to a :class:`Transceiver` instance, the :meth:`Transceiver.connect` method
    must with the :class:`Medium` instance as an argument."""

    def __init__(self):
        self._xcvrs: set[Transceiver] = set()

    def __repr__(self) -> str:
        # This is just a placeholder to prevent logs from being unwieldy. Subclasses
        # should override this method to provide more useful information.
        return f'<{self.__class__.__name__}>'

    def __getattr__(self, name: str) -> Any:
        """Really hammer the point home that the user should be calling connect() and
        disconnect() on a Transceiver instance, not a Medium instance."""
        if name in ('connect', 'disconnect'):
            raise AttributeError(
                f'{self.__class__.__name__} instances do not have a {name}() method. '
                f'Use {name}() of an applicable Transceiver instance instead.'
            )
        raise AttributeError(f'{self.__class__.__name__} has no attribute {name}')

    # region Abstract methods

    @abstractmethod
    def _subclass_connect(self, xcvr: Transceiver) -> Any:
        """Run class-specific connection logic and return any metadata to be stored in
        the :attr:`_xcvrs` dictionary alongside the :class:`Transceiver`"""
        raise NotImplementedError

    @abstractmethod
    def _subclass_disconnect(self, xcvr: Transceiver):
        """Run class-specific disconnection logic associated with the removal of a
        :class:`Transceiver` from the medium"""
        raise NotImplementedError

    # endregion

    # region Private methods

    def _connect(self, xcvr: Transceiver, **kwargs):
        """Connect a :class:`Transceiver` instance to the medium.

        :param xcvr: The :class:`Transceiver` instance to connect to the medium.

        NOTE: This method is called by :meth:`Transceiver.connect` and must not be
        called in any other contexts."""
        self._subclass_connect(xcvr, **kwargs)
        self._xcvrs.add(xcvr)

    def _disconnect(self, xcvr: Transceiver):
        """Disconnect a :class:`Transceiver` instance from the medium.

        :param xcvr: The :class:`Transceiver` instance to disconnect from the medium.

        NOTE: This method is called by :meth:`Transceiver.disconnect` and must not be
        called in any other contexts."""
        self._subclass_disconnect(xcvr)
        self._xcvrs.remove(xcvr)

    # endregion


class Transceiver(ABC):
    """A base class for all transceivers. A transceiver is a device that can send and
    receive data over a physical medium. In PyNet, a :class:`Transceiver` instance is
    responsible for connecting to a :class:`Medium` instance."""

    _supported_media: Sequence[Medium]
    _medium: Medium = None

    def __init_subclass__(cls, supported_media: Sequence[Medium]) -> None:
        bad_media = [m for m in supported_media if not issubclass(m, Medium)]
        if bad_media:
            raise TypeError(f'{bad_media} are not subclasses of {Medium}')

        cls._supported_media = supported_media

    def __repr__(self) -> str:
        # This is just a placeholder to prevent logs from being unwieldy. Subclasses
        # should override this method to provide more useful information.
        if self._medium:
            medium_str = f'medium={self._medium.__class__.__name__}'
        else:
            medium_str = 'disconnected'
        return f'<{self.__class__.__name__} ({medium_str})>'

    # region Properties

    @property
    @abstractmethod
    def location(self):
        """The location of the transceiver. Details of this are subclass-specific. For
        example, a 10Base5 transceiver's location may be a 1D position along a coaxial
        cable, whereas a WiFi transceiver may be in 3D space."""
        raise NotImplementedError

    @property
    def medium(self) -> Medium:
        return self._medium

    @medium.setter
    def medium(self, medium: Medium) -> None:
        self.connect(medium)

    # endregion

    # region Abstract methods

    @abstractmethod
    def _connect(self, medium: Medium, **kwargs):
        """Class-specific connection logic"""
        raise NotImplementedError

    @abstractmethod
    def _disconnect(self):
        """Class-specific disconnection logic"""
        raise NotImplementedError

    @abstractmethod
    def put(self, data: str):
        """Send data over the medium"""
        # TODO: data should probably be a bytes-like object
        raise NotImplementedError

    @abstractmethod
    def listen(self, timeout: int | float = None):
        """Receive data from the medium"""
        raise NotImplementedError

    # endregion

    # region Public methods

    def connect(self, new_medium: Medium, replace=False, **kwargs) -> None:
        if not new_medium:
            # :meth:`Transceiver.connect` could have been called via the setter method
            # with a `None` value. We want to let this happen without raising an error,
            # but we should log the re-routing for debugging purposes.
            log.debug('Connecting to non-existant medium. Assuming disconnect')
            self.disconnect()
            return

        # Make sure to check the validity of the incoming medium prior to
        # (potentially) disconnecting from the current medium.
        if (medium_cls := new_medium.__class__) not in self._supported_media:
            raise ValueError(f'Medium {medium_cls.__name__} not supported by {self!r}')

        # If the :class:`Transceiver` is already connected to a :class:`Medium`. Nothing
        # to do if this is the same medium being connected. Otherwise make sure to
        # disconnect from the previous medium (or raise an error if ``replace``` is
        # `False`).
        if current_medium := self._medium:
            if new_medium is current_medium:
                log.debug(f'{self!r} already connected to {new_medium!r}')
                return

            if replace:
                log.debug(f'Replacing {current_medium} with {new_medium!r}')
                self.disconnect()
            else:
                raise RuntimeError(
                    f'{self!r} already connected to {current_medium!r}. Use replace=True '
                    'to swap mediums or call disconnect() first'
                )

        log.debug(f'Connecting {self!r} to {new_medium!r}')

        # Make sure that both the subclass and the :class:`Medium` subclass run their
        # respective connection logic.
        # NOTE:
        # Order is important here. The medium will need to know the transceiver's
        # information in some cases.
        self._connect(new_medium, **kwargs)
        new_medium._connect(self, **kwargs)
        self._medium = new_medium

    def disconnect(self) -> None:
        if not (medium := self._medium):
            # Already disconnected
            return

        # No need to specify the medium here since :meth:`__repr__` will return the
        # medium detatils.
        log.debug(f'Disconnecting {self!r}')

        # Make sure that both the subclass and the :class:`Medium` subclass run their
        # respective disconnection logic.
        self._disconnect()
        medium._disconnect(self)
        self._medium = None

    # endregion
