#!/usr/bin/env python
from __future__ import annotations

from contextlib import contextmanager
from unittest import TestCase, main
from unittest.mock import call, patch

from pynet.physical.base import Transceiver, Medium
from pynet.testing import LogTestingMixin

BASE_TARGET = 'pynet.physical.base'


class MockMedium(Medium):
    """A helper class for testing the :class:`Medium` class. Defines the necessary
    abstract methods to instantiate the class."""

    def __init__(self, name: str = 'mock'):
        super().__init__()
        # A name is useful for test debugging.
        self.name = name

    def __repr__(self):
        return f'<{self.__class__.__name__} (name={self.name!r})>'

    def _subclass_connect(self, transceiver: MockTransceiver):
        pass

    def _subclass_disconnect(self, transceiver: MockTransceiver):
        pass


class MockTransceiver(Transceiver, supported_media=[MockMedium]):
    """A helper class for testing the :class:`Transceiver` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _connect(self, medium: MockMedium):
        pass

    def _disconnect(self):
        pass

    def put(self, data: str):
        pass

    def listen(self, timeout: int | float = None):
        pass


class TestTransceiver(TestCase, LogTestingMixin):
    @contextmanager
    def assertInBaseLogs(self, level, msgs, *args, **kwargs):
        with self.assertInLogs(BASE_TARGET, level, msgs, *args, **kwargs):
            yield

    @contextmanager
    def assertNotInBaseLogs(self, level, msgs, *args, **kwargs):
        with self.assertNotInLogs(BASE_TARGET, level, msgs, *args, **kwargs):
            yield

    def test__init_subclass__with_bad_media(self):
        with self.assertRaises(TypeError):

            class BadTransceiver(Transceiver, supported_media=[object]):
                pass

    def test__init_subclass__with_good_media(self):
        test_media = [Medium]

        class GoodTransceiver(Transceiver, supported_media=test_media):
            pass

        self.assertEqual(GoodTransceiver._supported_media, test_media)

    @patch.object(MockTransceiver, 'connect')
    def test_medium_setter_wraps_connect(self, connect_mock):
        transceiver = MockTransceiver()
        medium = MockMedium()
        transceiver.medium = medium

        connect_mock.assert_called_once_with(medium)

    def test_medium_getter(self):
        medium = MockMedium()
        transceiver = MockTransceiver()
        self.assertIsNone(transceiver.medium)

        transceiver._medium = medium
        self.assertEqual(transceiver.medium, medium)

    @patch.object(MockTransceiver, 'disconnect')
    def test_connecting_nothing_same_as_disconnecting(self, disconnect_mock):
        transceiver = MockTransceiver()
        transceiver.medium = MockMedium()

        with self.assertInBaseLogs(
            'DEBUG', 'Connecting to non-existant medium. Assuming disconnect'
        ):
            transceiver.connect(None)

        disconnect_mock.assert_called_once()

    def test_connecting_same_medium_does_nothing(self):
        medium_name = 'test'
        medium = MockMedium(name=medium_name)
        transceiver = MockTransceiver()
        transceiver.medium = medium

        with self.assertNotInBaseLogs(
            'DEBUG',
            'Connecting.*MockTransceiver',
            regex=True,
        ):
            transceiver.connect(medium)

    def test_connect_unsupported_medium_raises(self):
        transceiver = MockTransceiver()

        with self.assertRaisesRegex(ValueError, 'Medium.*not supported by.*Transceiver'):
            transceiver.connect(object())

    def test_connecting_without_replace_when_already_connected_raises(self):
        medium = MockMedium()
        transceiver = MockTransceiver()
        transceiver.medium = medium

        with self.assertRaisesRegex(RuntimeError, 'Transceiver.*already connected to'):
            transceiver.connect(MockMedium())

    @patch.object(MockTransceiver, '_disconnect')
    @patch.object(MockTransceiver, '_connect')
    def test_connecting_with_replace_when_already_connected(
        self, connect_mock, disconnect_mock
    ):
        # This actually tests both the connect and disconnect logic.
        first_name = 'first'
        second_name = 'second'
        first_medium = MockMedium(name=first_name)
        second_medium = MockMedium(name=second_name)
        transceiver = MockTransceiver()

        disconnected_str = '<MockTransceiver (disconnected)>'

        with self.assertInBaseLogs(
            'DEBUG',
            f'Connecting {disconnected_str} to <MockMedium (name={first_name!r})>',
        ):
            transceiver.connect(first_medium)

        with self.assertInBaseLogs(
            'DEBUG',
            [
                'Disconnecting <MockTransceiver (medium=MockMedium)>',
                f'Connecting {disconnected_str} to <MockMedium (name={second_name!r})>',
            ],
        ):
            transceiver.connect(second_medium, replace=True)

        connect_mock.assert_has_calls([call(first_medium), call(second_medium)])
        disconnect_mock.assert_called_once()
        self.assertEqual(transceiver._medium, second_medium)

    @patch.object(MockTransceiver, 'disconnect')
    def test_connecting_when_disconnected_does_not_disconnect(self, disconnect_mock):
        transceiver = MockTransceiver()
        medium = MockMedium()

        with self.assertNotInBaseLogs(
            'DEBUG', 'Disconnecting.*MockTransceiver', regex=True
        ):
            with self.assertInBaseLogs(
                'DEBUG',
                'Connecting.*MockTransceiver.*to.*MockMedium',
                regex=True,
            ):
                transceiver.connect(medium)

        disconnect_mock.assert_not_called()

    @patch.object(MockTransceiver, '_disconnect')
    def test_disconnect_called_when_already_disconnected(self, disconnect_mock):
        transceiver = MockTransceiver()

        with self.assertNotInBaseLogs(
            'DEBUG', 'Disconnecting.*MockTransceiver', regex=True
        ):
            transceiver.disconnect()

        disconnect_mock.assert_not_called()

    def test_successful_disconnect(self):
        medium = MockMedium()
        transceiver = MockTransceiver()
        transceiver.medium = medium

        with self.assertInBaseLogs('DEBUG', 'Disconnecting.*MockTransceiver', regex=True):
            transceiver.disconnect()

        self.assertIsNone(transceiver._medium)


class TestMedium(TestCase, LogTestingMixin):
    @patch.object(MockMedium, '_subclass_disconnect')
    @patch.object(MockMedium, '_subclass_connect')
    def test_metadata_processing_in_connect_and_disconnect(
        self, connect_mock, disconnect_mock
    ):
        medium = MockMedium()
        transceiver = MockTransceiver()

        connect_mock.return_value = metadata = {'test': 'value'}

        transceiver.connect(medium)

        connect_mock.assert_called_once_with(transceiver)
        self.assertEqual(metadata, medium._transceivers[transceiver])

        transceiver.disconnect()
        disconnect_mock.assert_called_once_with(transceiver)
        self.assertNotIn(transceiver, medium._transceivers)


if __name__ == '__main__':
    main()
