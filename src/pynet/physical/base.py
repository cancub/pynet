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
    connections or disconnections, nor does it send or receive data. In PyNet, it exists
    merely as a record-keeper of the :class:`Transceiver` instances that are connected to
    it and as a means of transporting data between those :class:`Transceiver` instances in
    a manner that simulates the physical medium.

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

    _transceivers: dict[Transceiver, Any] = {}

    def __repr__(self) -> str:
        # This is just a placeholder to prevent logs from being unwieldy. Subclasses
        # should override this method to provide more useful information.
        return f'<{self.__class__.__name__}>'

    @abstractmethod
    def _subclass_connect(self, transceiver: Transceiver) -> Any:
        """Run class-specific connection logic and return any metadata to be stored in
        the :attr:`_transceivers` dictionary alongside the :class:`Transceiver`"""
        raise NotImplementedError

    @abstractmethod
    def _subclass_disconnect(self, transceiver: Transceiver):
        """Run class-specific disconnection logic associated with the removal of a
        :class:`Transceiver` from the medium"""
        raise NotImplementedError

    def _connect(self, transceiver: Transceiver, **kwargs):
        """Connect a :class:`Transceiver` instance to the medium and optionally store
        metadata provided by the subclass.

        :param transceiver: The :class:`Transceiver` instance to connect to the medium.

        NOTE: This method is called by :meth:`Transceiver.connect` and must not be
        called in any other contexts."""
        self._transceivers[transceiver] = self._subclass_connect(transceiver, **kwargs)

    def _disconnect(self, transceiver: Transceiver):
        """Disconnect a :class:`Transceiver` instance from the medium.

        :param transceiver: The :class:`Transceiver` instance to disconnect from the
            medium.

        NOTE: This method is called by :meth:`Transceiver.disconnect` and must not be
        called in any other contexts."""
        self._subclass_disconnect(transceiver)

        del self._transceivers[transceiver]


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

    @property
    def medium(self) -> Medium:
        return self._medium

    @medium.setter
    def medium(self, medium: Medium) -> None:
        self.connect(medium)

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
