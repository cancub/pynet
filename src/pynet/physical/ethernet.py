"""
Ethernet physical layer devices.

TODO:
- implement other 10BASEX variants
- implement XBASE-T
- implement repeater
- implement hub
"""

import logging
from functools import cached_property
from math import ceil
from typing import Optional, Type

from .base import Medium, Transceiver
from .exceptions import InvalidLocationError, NoSuitableLocationsError

__all__ = ['Coaxial', 'RG_8U', 'TenBaseFiveTransceiver']

log = logging.getLogger(__name__)


# region Media


class Coaxial(Medium):
    length: int

    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = length


class RG_8U(Coaxial):
    """RG-8/U is a type of coaxial cable used in Ethernet networks. In order to prevent
    signal reflections at the end of the cable, a terminator is used, with this most
    commonly being a 50 ohm resistor that is connected via a male N-type connector.
    TODO:
    - simulate delays between nodes based on distances
    - simulate noise
    - simulate the cable breaking
    - simulate resistors at the ends of the cable
    - simulate signal reflections when resistors are not used
    - simulate reflections from multiple taps related to phase (2.5m)
    """

    _standard_interval: int = 2.5  # meters (IEEE 802.3-1985)

    def __init__(self, length: int, xcvr_type: Type[Transceiver]) -> None:
        super().__init__(length)

        # Any transceiver that is connected to the cable must be connected via a tap. It
        # therefor must have a ``length`` attribute.
        try:
            # Half the length of the transceiver plus a 20% buffer.
            xcvr_first_offset = xcvr_type.length * 1.2 / 2
        except AttributeError:
            raise TypeError(
                'Tranceivers must have a length class attribute to be connected to a '
                'RG_8U cable.'
            )

        self._xcvr_type: Type[Transceiver] = xcvr_type

        # The dict of locations where it is suitable to place a tap, with keys specifying
        # the locations in meters from the start of the cable and the values specifying
        # any :class:`Transceiver` that is occupying that location (either optimally or
        # suboptimally).
        # Mark the suitable locations for taps (maximizing space by starting from an
        # offset just slightly larger than half the length of the transceiver tap) and
        # show that nothing is connected at these locations.
        # NOTE:
        # This is based on the IEEE Standard 802.3-1985, which states that nodes must be
        # placed at intervals of 2.5 meters to prevent signal reflections between nodes
        # from being in phase.
        # Taps may be placed at any location on the cable, and a warning will be logged if
        # this location is not suitable for a tap.
        self._suitable_location_usage: dict[int, Optional[Transceiver]] = {
            i * self._standard_interval + xcvr_first_offset: None
            for i in range(ceil(length / self._standard_interval))
        }

    # region Properties

    @property
    def suitable_locations(self) -> list[int]:
        """All suitable locations on the cable where a tap can be placed."""
        return list(self._suitable_location_usage.keys())

    @property
    def available_locations(self) -> list[int]:
        """The subset of unused suitable locations on the cable where a tap can be
        placed."""
        return [loc for loc, used in self._suitable_location_usage.items() if not used]

    @cached_property
    def _xcvr_len(self) -> float:
        return self._xcvr_type.length

    @cached_property
    def _xcvr_half_len(self) -> float:
        return self._xcvr_len / 2

    # endregion

    # region Abstract method re-definitions

    def _subclass_connect(self, xcvr: Transceiver, **kwargs):
        # The transceiver has done the work to ensure that the location is has enough
        # space for a tap. Our job is to log a warning if the location is not suitable or
        # mark the location as being used if it is suitable.
        location = xcvr.location

        # Log a message if the location is not suitable for a tap.
        if location in self._suitable_location_usage:
            # Mark the location as being used and log a message to show that the location
            # makes sense.
            log.debug(f'{xcvr} now optimally occupying suitable location ({location} m)')
            self._suitable_location_usage[location] = xcvr
        else:
            warnings = [
                f'The provided location ({location} m) is not suitable for a tap. Some '
                'nodes may experience signal reflections.'
            ]
            # This location can still use up a suitable location if the transceiver would
            # overlap with another transceiver at this location. Again, keep in mind that
            # this method is only called if there is enough space for a tap.
            if nearest := self._get_overlapping_suitable_location(location):
                warnings.append(
                    f'Additionally, the provided location ({location} m) overlaps with a '
                    f'predefined suitable location ({nearest} m) and this location will '
                    'be marked as occupied.'
                )
                self._suitable_location_usage[nearest] = xcvr

            log.warning(' '.join(warnings))

    def _subclass_disconnect(self, xcvr: Transceiver):
        try:
            location_in_use = next(
                loc for loc, node in self._suitable_location_usage.items() if node is xcvr
            )
        except StopIteration:
            # The transceiver was not taking up a suitable location.
            return

        self._suitable_location_usage[location_in_use] = None
        log.debug(f'Disconnection has freed up a suitable location ({location_in_use} m)')

    # endregion

    # region Private methods

    def _get_nearest_xcvr(self, location: int) -> Optional[Transceiver]:
        """Find the nearest transceiver to the provided location.

        :param location: The location to use as the starting point for the search
        :return: The nearest transceiver
        """
        return min(
            self._xcvrs,
            key=lambda xcvr: abs(xcvr.location - location),
            default=None,
        )

    def _get_nearest_suitable_location(
        self, location: int = 0, filter_in_use: bool = True
    ) -> Optional[float]:
        """Find the nearest suitable location to the provided location.

        :param location: The location to use as the starting point for the search
        :param filter_in_use: If true, filter out locations that are already in use
        :return: The nearest suitable location
        """
        return min(
            self.available_locations if filter_in_use else self._suitable_location_usage,
            key=lambda loc: abs(loc - location),
            default=None,
        )

    def _get_overlapping_suitable_location(self, location: int) -> Optional[float]:
        """Check if the provided location overlaps with a suitable location (regardless of
        whether or not that suitable location is in use) and return the overlapping
        suitable location if it does.

        :param location: The location to check
        :return: The overlapping suitable location or None if there is no overlap.
        """
        nearest = self._get_nearest_suitable_location(location, filter_in_use=False)
        if abs(nearest - location) < self._xcvr_len:
            return nearest
        return None

    # endregion

    # region Public methods

    def check_valid_location(self, location: int):
        start = location - self._xcvr_half_len
        end = location + self._xcvr_half_len

        # Would there be enough space between the edges of this transceiver and the ends
        # of the cable?
        if start < 0 or end > self.length:
            raise InvalidLocationError(
                f'The location ({location} m) would cause one or both ends of the '
                f'transceiver to extend past the end of the {self.length}-meter cable '
                f'(range=[{start}, {end}]).'
            )

        # Would there be enough space between the end of this transceiver and the end of
        # the nearest existing transceiver?
        nearest_node = self._get_nearest_xcvr(location)
        if nearest_node and abs(nearest_node.location - location) < self._xcvr_len:
            raise InvalidLocationError(
                f'The location ({location} m) is too close to an existing node '
                f'on the cable at {nearest_node.location} m.'
            )

    # endregion


# endregion

# region Transceivers


class TenBaseFiveTransceiver(Transceiver, supported_media=[RG_8U]):
    """The OG Ethernet transceiver. Connected in a bus topology on an RG-8/U cable.
    Nodes are connected along the cable via
    `vampire taps<https://en.wikipedia.org/wiki/Vampire_tap>`_.

    Resources:
    - https://en.wikipedia.org/wiki/10BASE5
    - https://www.networkencyclopedia.com/10base5/

    TODO:
    - implement clock
    - implement manchester encoding/decoding according to IEEE 802.3
    """

    # The amount of length-wise real-estate (in meters) taken up by the transceiver on the
    # cable. This is used to determine if there is enough space between this node and the
    # next node.
    # NOTE: this is just a ballpark estimate. Might need to revise later.
    length: int = 0.15

    # The center of the transceiver on the cable. That is, if the tap is connected at 10m,
    # then the ends of the transceiver will extend to 9.925m and 10.075m.
    _location: int

    # region Redefined Methods

    def _connect(self, coax: RG_8U, location: int = None):
        """Connect to an RG-8/U cable via vampire connection, checking if there is
        enough space between this transceiver and the next transceiver on the cable.

        :param coax: The :class:`RG_8U` cable to which this transceiver will connect
        :param location: The location (in meters) from the start of the cable. If not
            provided, it is assumed that the is connecting the transceiver the optimal
            location from other transceivers / the ends of the cable.

        TODO:
        - add the option of providing a range of locations in the form of a 2-tuple of
            ints. This is to allow for the idea of transceivers existing in offices that
            only have access to a certain range of the cable.
        """
        if location is None:
            # Check the coax cable markers for the next suitable location.
            if (location := coax._get_nearest_suitable_location()) is None:
                # There are no suitable locations on the cable. Raise an exception to
                # alert the caller that they need to manually select a location.
                raise NoSuitableLocationsError(
                    'There are no suitable locations remaining on the cable. Please '
                    'manually select a location by calling connect with the ``location`` '
                    'argument.'
                )

        # Make sure the caller isn't doing anything silly like connecting past the ends of
        # the cable or overlapping with another transceiver.
        coax.check_valid_location(location)

        log.debug(f'{self} connected to {coax} at {location} m')
        self._location = location

    def _disconnect(self):
        self._location = None

    @property
    def location(self):
        return self._location

    def put(self, data):
        pass

    def listen(self, timeout: int | float = None):
        pass

    # endregion


# endregion
