import logging
from abc import ABC, abstractmethod
from typing import Sequence

__all__ = ['Medium', 'Transceiver']
log = logging.getLogger(__name__)


class Medium(ABC):
    pass


class Transceiver(ABC):
    supported_media: Sequence[Medium]
    _medium: Medium = None

    def __init_subclass__(cls, supported_media: Sequence[Medium]) -> None:
        bad_media = [m for m in supported_media if not issubclass(m, Medium)]
        if bad_media:
            raise TypeError(f'{bad_media} are not subclasses of {Medium}')

        cls.supported_media = supported_media

    def __init__(self, medium: Medium = None) -> None:
        if medium:
            self.connect(medium)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} (medium={self._medium})'

    @property
    def medium(self) -> Medium:
        return self._medium

    @medium.setter
    def medium(self, medium: Medium) -> None:
        self.connect(medium)

    def connect(self, medium: Medium, replace=False) -> None:
        # Connecting nothing is the same as disconnecting.
        if not medium:
            self.disconnect()
            return

        if medium is self._medium:
            # Already connected to the same medium
            return

        # Make sure to check the validity of the incoming medium prior to
        # (potentially) disconnecting from the current medium.
        if medium.__class__ not in self.supported_media:
            raise ValueError(f'Medium {medium} not supported by {self}')

        # Validate medium prior to connecting.
        if self._medium is not None:
            if replace:
                # The caller has specified that they want to replace the current
                # medium with the new one, which is fine. We just need to
                # perform the proper disconnect logic first.
                self.disconnect()
            else:
                raise RuntimeError(
                    f'{self} already connected to {self._medium}'
                )

        log.debug(f'{self} connecting to {medium}')

        self._connect(medium)
        self._medium = medium

    def _connect(self, medium: Medium):
        """Optional class-specific connection logic"""
        pass

    def disconnect(self) -> None:
        if not self._medium:
            # Already disconnected
            return

        log.debug(f'{self} disconnecting from {self._medium}')
        self._disconnect()
        self._medium = None

    def _disconnect(self):
        """Optional class-specific disconnection logic"""
        pass

    @abstractmethod
    def send(self, data: str):
        """Send data over the medium"""
        # TODO: data should probably be a bytes-like object
        pass

    @abstractmethod
    def receive(self, timeout: int | float = None):
        """Receive data from the medium"""
        pass
