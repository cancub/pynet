#! /usr/bin/env python
from contextlib import contextmanager
from unittest import TestCase, main

from pynet.physical.base import Transceiver
from pynet.physical.ethernet import Thicknet, TenBaseFiveTransceiver
from pynet.physical.exceptions import InvalidLocationError, NoSuitableLocationsError
from pynet.testing import LogTestingMixin

ETH_TARGET = 'pynet.physical.ethernet'


class EthernetTestCase(TestCase, LogTestingMixin):
    def setUp(self):
        super().setUp()
        self.xcvr = TenBaseFiveTransceiver()
        self.coax = Thicknet(length=6, xcvr_type=TenBaseFiveTransceiver)

    @contextmanager
    def assertInEthLogs(self, level, msgs, *args, **kwargs):
        with self.assertInLogs(ETH_TARGET, level, msgs, *args, **kwargs):
            yield

    @contextmanager
    def assertNotInEthLogs(self, level, msgs, *args, **kwargs):
        with self.assertNotInLogs(ETH_TARGET, level, msgs, *args, **kwargs):
            yield


class TestThicknet(EthernetTestCase):

    # region Dunders

    def test_init(self):
        length = 8
        coax = Thicknet(length=length, xcvr_type=TenBaseFiveTransceiver)

        self.assertEqual(coax.length, length)
        self.assertEqual(coax._xcvrs, set())
        self.assertEqual(coax._xcvr_type, TenBaseFiveTransceiver)
        self.assertEqual(
            coax._suitable_location_usage,
            {0.09: None, 2.59: None, 5.09: None, 7.59: None},
        )

    def test_init_with_xcvr_with_no_length(self):
        class BadTransceiver(Transceiver, supported_media=[Thicknet]):
            pass

        with self.assertRaisesRegex(
            TypeError,
            (
                'Tranceivers must have a length class attribute to be connected to a '
                'Thicknet cable.'
            ),
        ):
            Thicknet(length=1, xcvr_type=BadTransceiver)

    # endregion

    # region Properties

    def test_suitable_locations(self):
        self.coax._suitable_location_usage[2.59] = True
        self.assertEqual(self.coax.suitable_locations, [0.09, 2.59, 5.09])

    def test_available_locations(self):
        self.coax._suitable_location_usage[2.59] = True
        self.assertEqual(self.coax.available_locations, [0.09, 5.09])

    def test_xcvr_lengths(self):
        self.assertEqual(self.coax._xcvr_len, TenBaseFiveTransceiver.length)
        self.assertEqual(self.coax._xcvr_half_len, TenBaseFiveTransceiver.length / 2)

    # endregion

    # region Abstract Method redefinitions

    def test_subclass_connect_at_suitable_location(self):
        # Automatically use the first available location.
        with self.assertInEthLogs(
            'DEBUG', f'{self.xcvr} now optimally occupying suitable location (0.09 m)'
        ):
            self.xcvr.connect(self.coax)

        self.assertEqual(self.coax._xcvrs, {self.xcvr})
        self.assertEqual(
            self.coax._suitable_location_usage, {0.09: self.xcvr, 2.59: None, 5.09: None}
        )

    def test_subclass_connect_at_unsuitable_location_no_overlap(self):
        suitable_locations = self.coax.suitable_locations
        location = (suitable_locations[0] + suitable_locations[1]) / 2

        # We don't take out a suitable location.
        with self.assertNotInEthLogs('WARNING', f'({location} m) overlaps with'):
            with self.assertInEthLogs(
                'WARNING', f'({location} m) is not suitable for a tap'
            ):
                self.xcvr.connect(self.coax, location=location)

        # Confirm that all suitable locations are still available.
        self.assertEqual(
            len(self.coax.available_locations), len(self.coax.suitable_locations)
        )

    def test_subclass_connect_at_unsuitable_location_with_overlap(self):
        suitable_to_overlap = self.coax.suitable_locations[1]
        location = suitable_to_overlap + self.xcvr.length / 2

        with self.assertInEthLogs(
            'WARNING',
            [
                f'({location} m) is not suitable for a tap',
                (
                    f'({location} m) overlaps with a predefined suitable location '
                    f'({suitable_to_overlap} m)'
                ),
            ],
        ):
            self.xcvr.connect(self.coax, location=location)

        # Confirm that the overlapped suitable location is now occupied by the
        # transceiver.
        self.assertEqual(
            self.coax._suitable_location_usage, {0.09: None, 2.59: self.xcvr, 5.09: None}
        )

    def test_subclass_disconnect_without_overlap_does_not_free_up(self):
        suitable_locations = self.coax.suitable_locations
        location = (suitable_locations[0] + suitable_locations[1]) / 2

        self.xcvr.connect(self.coax, location=location)

        with self.assertNotInEthLogs(
            'DEBUG', 'Disconnection has freed up a suitable location'
        ):
            self.xcvr.disconnect()

    def test_subclass_disconnect_with_overlap_frees_up(self):
        suitable_to_overlap = self.coax.suitable_locations[1]
        location = suitable_to_overlap + self.xcvr.length / 2

        self.xcvr.connect(self.coax, location=location)

        with self.assertInEthLogs(
            'DEBUG', 'Disconnection has freed up a suitable location (2.59 m)'
        ):
            self.xcvr.disconnect()

        # Confirm that the overlapped suitable location is now available.
        self.assertEqual(
            self.coax._suitable_location_usage, {0.09: None, 2.59: None, 5.09: None}
        )

    # endregion

    # region Private Methods

    def test_get_nearest_xcvr_no_xcvrs(self):
        self.assertIsNone(self.coax._get_nearest_xcvr(0))

    def test_get_nearest_xcvr(self):
        other_xcvr = TenBaseFiveTransceiver()
        self.xcvr.connect(self.coax)
        other_xcvr.connect(self.coax)

        self.assertIs(self.coax._get_nearest_xcvr(1), self.xcvr)

    def test_get_nearest_suitable_location(self):
        # Fill up all the suitable locations.
        while True:
            try:
                TenBaseFiveTransceiver().connect(self.coax)
            except NoSuitableLocationsError:
                break

        # All the good spots are taken.
        self.assertFalse(self.coax.available_locations)

        # We cannot see any suitable locations if we filter in-use locations.
        self.assertIsNone(self.coax._get_nearest_suitable_location(0, filter_in_use=True))

        # We still get the expected suitable locations without filtering.
        self.assertEqual(
            self.coax._get_nearest_suitable_location(0, filter_in_use=False), 0.09
        )
        self.assertEqual(
            self.coax._get_nearest_suitable_location(2, filter_in_use=False), 2.59
        )

    def test_get_overlapping_suitable_location(self):
        self.assertEqual(self.coax._get_overlapping_suitable_location(0), 0.09)
        self.assertIsNone(self.coax._get_overlapping_suitable_location(1))

    def test_check_valid_location_too_close_to_end(self):
        # Too close to the start.
        with self.assertRaisesRegex(
            InvalidLocationError,
            r'The location \(0 m\) .* the 6-meter cable \(range=\[-0.075, 0.075\]\).',
        ):
            self.coax.check_valid_location(0)

        # Too close to the end.
        with self.assertRaisesRegex(
            InvalidLocationError,
            r'The location \(6 m\) .* the 6-meter cable \(range=\[5.925, 6.075\]\).',
        ):
            self.coax.check_valid_location(6)

    def test_check_valid_location_too_close_to_other_xcvr(self):
        self.xcvr.connect(self.coax)

        with self.assertRaisesRegex(
            InvalidLocationError, r'\(0.15 m\) is too close .* at 0.09 m.'
        ):
            self.coax.check_valid_location(0.15)

    def test_check_valid_location_is_good(self):
        self.coax.check_valid_location(0.15)

    # endregion


class TestTenBaseFiveTransceiver(EthernetTestCase):

    # region Abstract Method redefinitions

    def test_connect_gets_next_suitable_location_by_default(self):
        coax = self.coax
        locations = coax.suitable_locations

        def connect_and_check(expected_index):
            xcvr = TenBaseFiveTransceiver()
            loc = locations[expected_index]

            with self.assertInEthLogs('DEBUG', f'{xcvr} connected to {coax} at {loc} m'):
                xcvr.connect(coax)

            self.assertEqual(xcvr.location, loc)

            return xcvr

        xcvr1 = connect_and_check(0)
        connect_and_check(1)

        # Disconnect the first transceiver and observe that connecting a new transceiver
        # will use the freed-up location.
        xcvr1.disconnect()

        connect_and_check(0)

    # endregion


if __name__ == '__main__':
    main()
