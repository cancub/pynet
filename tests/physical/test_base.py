#!/usr/bin/env python
from unittest import TestCase, main
from unittest.mock import call, patch

from pynet.physical.base import Transceiver, Medium
from pynet.testing import LogTestingMixin

BASE_TARGET = 'pynet.physical.base'


class MockTransceiver(Transceiver, supported_media=[Medium]):
    def send(self, data: str):
        pass

    def receive(self, timeout: int | float = None):
        pass


class TestTransceiver(TestCase, LogTestingMixin):
    def test__init_subclass__with_bad_media(self):
        with self.assertRaises(TypeError):

            class BadTransceiver(Transceiver, supported_media=[object]):
                pass

    def test__init_subclass__with_good_media(self):
        test_media = [Medium]

        class GoodTransceiver(Transceiver, supported_media=test_media):
            pass

        self.assertEqual(GoodTransceiver.supported_media, test_media)

    @patch.object(MockTransceiver, 'connect')
    def test_init_with_and_without_medium(self, connect_mock):
        for medium in (None, Medium()):
            with self.subTest(medium=medium):
                MockTransceiver(medium=medium)

                if medium:
                    connect_mock.assert_called_once_with(medium)
                else:
                    connect_mock.assert_not_called()

            connect_mock.reset_mock()

    @patch.object(MockTransceiver, 'connect')
    def test_medium_setter_wraps_connect(self, connect_mock):
        transceiver = MockTransceiver()
        medium = Medium()
        transceiver.medium = medium

        connect_mock.assert_called_once_with(medium)

    def test_medium_getter(self):
        medium = Medium()
        transceiver = MockTransceiver()
        self.assertIsNone(transceiver.medium)

        transceiver._medium = medium
        self.assertEqual(transceiver.medium, medium)

    @patch.object(MockTransceiver, 'disconnect')
    def test_connecting_nothing_same_as_disconnecting(self, disconnect_mock):
        transceiver = MockTransceiver(medium=Medium())
        transceiver.connect(None)

        disconnect_mock.assert_called_once()

    def test_connecting_same_medium_does_nothing(self):
        medium = Medium()
        transceiver = MockTransceiver(medium=medium)

        with self.assertNotInLogs(BASE_TARGET, 'DEBUG', 'connecting to'):
            transceiver.connect(medium)

    def test_connect_unsupported_medium_raises(self):
        transceiver = MockTransceiver()

        with self.assertRaisesRegex(ValueError, 'not supported'):
            transceiver.connect(object())

    def test_connecting_without_replace_when_already_connected_raises(self):
        medium = Medium()
        transceiver = MockTransceiver(medium=medium)

        with self.assertRaisesRegex(RuntimeError, 'already connected to'):
            transceiver.connect(Medium())

    @patch.object(MockTransceiver, '_disconnect')
    @patch.object(MockTransceiver, '_connect')
    def test_connecting_with_replace_when_already_connected(
        self, connect_mock, disconnect_mock
    ):
        # This actually tests both the connect and disconnect logic.
        first_medium = Medium()
        second_medium = Medium()

        expected_logs = [
            f'connecting to {first_medium}',
            'disconnecting from',
            f'connecting to {second_medium}',
        ]
        with self.assertInLogs(BASE_TARGET, 'DEBUG', expected_logs):
            transceiver = MockTransceiver(medium=first_medium)
            transceiver.connect(second_medium, replace=True)

        connect_mock.assert_has_calls(
            [call(first_medium), call(second_medium)]
        )
        disconnect_mock.assert_called_once()
        self.assertEqual(transceiver._medium, second_medium)

    @patch.object(MockTransceiver, 'disconnect')
    def test_connecting_when_disconnected_does_not_disconnect(
        self, disconnect_mock
    ):
        transceiver = MockTransceiver()
        medium = Medium()

        with self.assertNotInLogs(BASE_TARGET, 'DEBUG', 'disconnecting from'):
            with self.assertInLogs(
                BASE_TARGET, 'DEBUG', f'connecting to {medium}'
            ):
                transceiver.connect(medium)

        disconnect_mock.assert_not_called()

    @patch.object(MockTransceiver, '_disconnect')
    def test_disconnect_called_when_already_disconnected(
        self, disconnect_mock
    ):
        transceiver = MockTransceiver()

        with self.assertNotInLogs(BASE_TARGET, 'DEBUG', 'disconnecting from'):
            transceiver.disconnect()

        disconnect_mock.assert_not_called()

    def test_successful_disconnect(self):
        medium = Medium()
        transceiver = MockTransceiver(medium=medium)

        with self.assertInLogs(
            BASE_TARGET, 'DEBUG', f'disconnecting from {medium}'
        ):
            transceiver.disconnect()

        self.assertIsNone(transceiver._medium)


if __name__ == '__main__':
    main()
