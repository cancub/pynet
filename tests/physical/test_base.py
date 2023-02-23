#!/usr/bin/env python
from __future__ import annotations

from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event, Lock
from unittest import TestCase, main
from unittest.mock import Mock, call, patch

from pynet.physical.base import Medium, Transceiver
from pynet.physical.constants import CLOSE_MSG, Responses
from pynet.physical.exceptions import (
    ConnectionError,
    NoMediumError,
    ProcessNotRunningError,
)
from pynet.testing import LogTestingMixin, ProcessBuilderMixin

BASE_TARGET = 'pynet.physical.base'

# region Helper Classes


class MockMedium(Medium, dimensionality=1):
    """A helper class for testing the :class:`Medium` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _connect(self, *args, **kwargs) -> int:
        with self._receivers_lock:
            return len(self._receivers)

    def _process_transmission(self, data, *args, **kwargs):
        return data


class MockTransceiver(Transceiver, supported_media=[MockMedium]):
    """A helper class for testing the :class:`Transceiver` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _process_incoming_data(self, data, *args, **kwargs):
        return data

    def _process_outgoing_data(self, data, *args, **kwargs):
        return data


# endregion

# region Tests


class TestPHYBase(TestCase, ProcessBuilderMixin, LogTestingMixin):
    medium_cls: type[Medium] = MockMedium
    xcvr_cls: type[Transceiver] = MockTransceiver
    log_target: str = BASE_TARGET

    def build_unique_mock(self, *names):
        # Prevent cross-talk between unit tests by explicitly setting the attributes
        # under test as brand-new mocks.
        return Mock(**{name: Mock() for name in names})


class TestMedium(TestPHYBase):
    def _run_monitor_connections(self, medium):
        medium._monitor_connections(
            medium._connection_queue,
            medium._receivers,
            medium._receivers_lock,
            medium._stop_event,
        )

    # region Dunders

    def test__init_subclass__(self):
        for dimensionality in (1, 3):
            with self.subTest(dimensionality=dimensionality):

                class DummyMedium(Medium, dimensionality=dimensionality):
                    pass

                self.assertEqual(DummyMedium._dimensionality, dimensionality)

        # Dimensionality cannot be anything other than 1 or 3
        for dimensionality in (0, 2, 4):
            with self.subTest(dimensionality=dimensionality):
                with self.assertRaisesRegex(
                    ValueError, '`dimensionality` must be 1 or 3'
                ):

                    class DummyMedium(Medium, dimensionality=dimensionality):
                        pass

    def test__new__(self):
        m1 = self.build_medium()
        self.assertEqual(Medium._instances[MockMedium, m1.name], m1)

        self.assertEqual(m1, self.build_medium(name=m1.name))

        m2 = self.build_medium()

        self.assertEqual(Medium._instances[MockMedium, m2.name], m2)
        self.assertNotEqual(m1, m2)

    @patch.object(MockMedium, 'start')
    def test__init__(self, mock_start):
        for auto_start in (True, False):
            with self.subTest(auto_start=auto_start):
                medium = self.build_medium(auto_start=auto_start)

                self.assertIsInstance(medium._tx_ingress_queue, Queue)
                self.assertEqual(medium._receivers, {})
                self.assertIsInstance(medium._receivers_lock, Lock)
                self.assertIsInstance(medium._connection_queue, Queue)
                self.assertIsInstance(medium._stop_event, Event)

                if auto_start:
                    mock_start.assert_called_once()
                else:
                    mock_start.assert_not_called()

                self.assertEqual(Medium._instances[MockMedium, medium.name], medium)

            mock_start.reset_mock()

    def test__repr__(self):
        medium = self.build_medium()
        self.assertEqual(
            repr(medium), f'<MockMedium: name={medium.name!r}, pid=None, running=False>'
        )

        medium.start()
        pid = medium.pid
        self.assertIsNotNone(pid)
        self.assertEqual(
            repr(medium), f'<MockMedium: name={medium.name!r}, pid={pid}, running=True>'
        )

        medium.stop()
        self.assertEqual(
            repr(medium), f'<MockMedium: name={medium.name!r}, pid={pid}, running=False>'
        )

    # endregion

    def test_lifecycle(self):
        for auto_start in (True, False):
            with self.subTest(auto_start=auto_start):
                medium = self.build_medium(auto_start=auto_start)
                self.assertEqual(Medium._instances[MockMedium, medium.name], medium)

                self.assertEqual(medium.is_alive(), auto_start)

                if not auto_start:
                    medium.start()

                self.assertIsNotNone(medium.pid)
                self.assertIsNotNone(medium._popen)
                self.assertTrue(medium.is_alive())

                medium.stop()
                self.assertIsNone(Medium._instances.get((MockMedium, medium.name)))
                self.assertFalse(medium.is_alive())

    # region run() tests

    @patch(f'{BASE_TARGET}.Thread', autospec=True)
    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch.object(MockMedium, '_process_medium')
    def test_run(self, process_mock, EventMock, ThreadMock):
        medium = self.build_medium()
        medium._stop_event.is_set.side_effect = [False, True]
        mock_thread = ThreadMock.return_value

        with self.assertInTargetLogs(
            'DEBUG',
            [f'Starting medium process ({medium.pid})', f'{medium}: shutting down'],
        ):
            medium.run()

        ThreadMock.assert_called_once_with(
            target=medium._monitor_connections,
            args=(
                medium._connection_queue,
                medium._receivers,
                medium._receivers_lock,
                medium._stop_event,
            ),
        )

        process_mock.assert_called_once()
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_disconnect')
    @patch.object(MockMedium, '_connect')
    def test_monitor_connections_add_connection_subclass_connect_error(
        self, connect_mock, disconnect_mock, *mocks
    ):
        medium = self.build_medium()
        mock_conn = Mock()
        kwargs = {'foo': 'bar'}

        # Allow for one loop to perform a connection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the new connection.
        medium._connection_queue.get.return_value = (mock_conn, None, kwargs)

        # Make the connection fail.
        err_str = 'Boom!'
        connect_mock.side_effect = err = Exception(err_str)

        # Use this test to check the debug logs as well.
        with self.assertInTargetLogs('ERROR', f'{medium}: connection failed: {err_str}'):
            self._run_monitor_connections(medium)

        # Confirm that there was one connection and no disconnection.
        connect_mock.assert_called_once_with(**kwargs)
        mock_conn.send.assert_called_once_with((Responses.ERROR, err))
        disconnect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_disconnect')
    @patch.object(MockMedium, '_connect')
    def test_monitor_connections_add_connection_success(
        self, connect_mock, disconnect_mock, *mocks
    ):
        medium = self.build_medium()
        location = 123
        mock_conn = Mock()
        kwargs = {'foo': 'bar'}

        # Allow for one loop to perform a connection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the new connection.
        medium._connection_queue.get.return_value = (mock_conn, None, kwargs)
        connect_mock.return_value = location

        # Use this test to check the debug logs as well.
        with self.assertInTargetLogs(
            'DEBUG',
            ['Starting connection worker thread', 'Connection worker thread exiting'],
        ):
            self._run_monitor_connections(medium)

        # Confirm that there was one connection and no disconnection.
        connect_mock.assert_called_once_with(**kwargs)
        mock_conn.send.assert_called_once_with((Responses.OK, location))
        self.assertEqual(medium._receivers[location], mock_conn)

        disconnect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_disconnect')
    @patch.object(MockMedium, '_connect')
    def test_monitor_connections_remove_connection_subclass_disconnect_error(
        self, connect_mock, disconnect_mock, *mocks
    ):
        medium = self.build_medium()
        location = 123

        medium._receivers[location] = mock_conn = Mock()

        # Allow for one loop to perform a disconnection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the new connection.
        medium._connection_queue.get.return_value = (None, location, None)

        # Make the disconnection fail.
        err_str = 'Boom!'
        disconnect_mock.side_effect = Exception(err_str)

        with self.assertInTargetLogs(
            'ERROR',
            (
                f'{medium}: subclass disconnection failed: {err_str}. Continuing with '
                'removal.'
            ),
        ):
            self._run_monitor_connections(medium)

        # Confirm that there was one disconnection and no connection.
        disconnect_mock.assert_called_once_with(location)
        mock_conn.close.assert_called_once()
        self.assertEqual(medium._receivers, {})

        connect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_disconnect')
    @patch.object(MockMedium, '_connect')
    def test_monitor_connections_remove_connection_success(
        self, connect_mock, disconnect_mock, *mocks
    ):
        medium = self.build_medium()
        location = 123

        medium._receivers[location] = mock_conn = Mock()

        # Allow for one loop to perform a disconnection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the new connection.
        medium._connection_queue.get.return_value = (None, location, None)

        self._run_monitor_connections(medium)

        # Confirm that there was one disconnection and no connection.
        disconnect_mock.assert_called_once_with(location)
        mock_conn.close.assert_called_once()
        self.assertEqual(medium._receivers, {})

        connect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    def test_monitor_connections_got_bad_connection_request(self, *mocks):
        medium = self.build_medium()

        # Allow for one loop to perform a get and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        medium._connection_queue.get.return_value = bad_conn_details = 666

        with self.assertInTargetLogs(
            'ERROR', f'{medium}: unexpected connection format: {bad_conn_details}'
        ):
            self._run_monitor_connections(medium)

    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    def test_process_medium_close_thread_received(self, *mocks):
        medium = self.build_medium()
        medium._tx_ingress_queue.get.return_value = CLOSE_MSG

        medium._process_medium()

    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    def test_process_medium_bad_transmission(self, *mocks):
        medium = self.build_medium()
        medium._tx_ingress_queue.get.return_value = bad_tx = 666

        with self.assertInTargetLogs(
            'ERROR',
            (
                f'{medium}: invalid transmission received ({bad_tx}). Format must be '
                '(location, data)'
            ),
        ):
            medium._process_medium()

    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_process_transmission', autospec=True)
    def test_process_medium_transmission_error(self, process_transmission_mock, *mocks):
        medium = self.build_medium()

        src_loc = 123
        src_conn = Mock()

        dest_loc = 456
        dest_conn = Mock()

        data = 'hello'
        medium._tx_ingress_queue.get.return_value = (data, src_loc)

        exc_str = 'A bad thing happened'
        process_transmission_mock.side_effect = Exception(exc_str)

        # Start by checking that the source is ignored.
        medium._receivers = receivers = {src_loc: src_conn}

        medium._process_medium()
        process_transmission_mock.assert_not_called()

        # Now add a viable destination such that the processing of the data is attempted.
        receivers[dest_loc] = dest_conn

        with self.assertInTargetLogs(
            'ERROR',
            (
                f'{medium}: error processing transmission ({data=}, {src_loc=}, '
                f'{dest_loc=}): {exc_str}'
            ),
        ):
            medium._process_medium()

        process_transmission_mock.assert_called_with(medium, data, src_loc, dest_loc)
        dest_conn.send.assert_not_called()

    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_process_transmission', autospec=True)
    def test_process_medium_pipe_error(self, process_transmission_mock, *mocks):
        medium = self.build_medium()

        src_loc = 123
        dest_loc = 456
        dest_conn = Mock()

        data = 'hello'
        medium._tx_ingress_queue.get.return_value = (data, src_loc)

        medium._receivers = {src_loc: None, dest_loc: dest_conn}
        out_data = process_transmission_mock.return_value

        exc_str = 'A bad thing happened'
        dest_conn.send.side_effect = Exception(exc_str)

        with self.assertInTargetLogs(
            'ERROR',
            f'{medium}: error sending data={out_data!r} to receiver at {dest_loc=}: {exc_str}',
        ):
            medium._process_medium()

        dest_conn.send.assert_called_with(out_data)

    @patch.object(MockMedium, '_disconnect')
    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_connect')
    @patch.object(MockMedium, '_process_transmission', autospec=True)
    def test_monitor_connection_process_medium_integration(
        self, process_transmission_mock, connect_mock, QueueMock, *mocks
    ):
        # Simulate a sequence of:
        # 1. A connection request.
        # 2. A transmission sent from a source and received by a destination.
        # 3. A disconnection request.
        # 4. A transmission sent from a source and _not_ received by a destination.
        tx_get_mock = Mock()
        conn_get_mock = Mock()
        QueueMock.side_effect = [tx_get_mock, conn_get_mock]

        medium = self.build_medium()

        src_loc = 123
        src_conn = Mock()
        dest_loc = 456
        dest_conn = Mock()

        medium._receivers = {src_loc: src_conn}

        data = 'hello'
        tx_get_mock.get.return_value = (data, src_loc)
        # An identity function is sufficient for this test.
        process_transmission_mock.return_value = data

        # Just to be sure, confirm that the destination connection doesn't get anything
        # sent at this point.
        medium._process_medium()
        dest_conn.send.assert_not_called()

        # Configure a connection request.
        medium._stop_event.is_set.side_effect = [False, True]
        conn_get_mock.get.return_value = (dest_conn, None, {})
        connect_mock.return_value = dest_loc

        # Add the destination to the receivers.
        self._run_monitor_connections(medium)
        dest_conn.send.assert_called_once_with((Responses.OK, dest_loc))
        dest_conn.reset_mock()

        # Send the transmission another time. This time. the destination should receive
        # it.
        medium._process_medium()
        dest_conn.send.assert_called_once_with(data)

        # Configure a disconnection request.
        medium._stop_event.is_set.side_effect = [False, True]
        conn_get_mock.get.return_value = (None, dest_loc, {})

        # Remove the destination from the receivers.
        self._run_monitor_connections(medium)

        # Send the transmission another time. This time, the destination should not
        # receive it.
        medium._process_medium()

    # endregion

    # region Public Methods

    @patch(f'{BASE_TARGET}.Process.terminate', autospec=True)
    @patch.object(MockMedium, 'join')
    def test_stop_in_theory(self, join_mock, terminate_mock):
        medium = self.build_medium()
        medium._stop_event = stop_event = Mock()
        medium._connection_queue = conn_queue = Mock()
        medium._tx_ingress_queue = tx_queue = Mock()
        medium.is_alive = Mock(return_value=True)

        medium.stop()

        stop_event.set.assert_called_once()
        conn_queue.put.assert_called_once_with(CLOSE_MSG)
        tx_queue.put.assert_called_once_with(CLOSE_MSG)

        join_mock.assert_called_once()
        terminate_mock.assert_called_once_with(medium)

        self.assertIsNone(Medium._instances.get((TestMedium, medium.name)))

        # Calling stop() again should not raise an exception.
        medium.stop()

    def test_stop_in_practice(self):
        medium = self.build_medium(auto_start=True)

        self.assertFalse(medium._stop_event.is_set())

        medium.stop()
        self.assertIsNone(Medium._instances.get((TestMedium, medium.name)))
        self.assertTrue(medium._stop_event.is_set())

        # We can still call stop a second time without raising an exception.
        medium.stop()

    def test_transmit(self):
        medium = self.build_medium()
        medium._tx_ingress_queue = tx_queue = Mock()

        data = 'hello'
        src_loc = 123
        medium.transmit(data, src_loc)

        tx_queue.put.assert_called_once_with((data, src_loc))

    # endregion


class TestTransceiver(TestPHYBase):
    def setUp(self):
        super().setUp()

        self.medium_mock = self.build_medium(mocked=True)

    # region Dunders

    def test__init_subclass__with_bad_media_types(self):
        with self.assertRaises(TypeError):

            class BadTransceiver(Transceiver, supported_media=[object]):
                pass

    def test__init_subclass__with_bad_media_dimensionalities(self):
        # Valid dimensionalities, but they differ and that's not allowed.
        class DummyMedium1(Medium, dimensionality=1):
            pass

        class DummyMedium2(Medium, dimensionality=3):
            pass

        with self.assertRaisesRegex(
            ValueError,
            '`supported_media` must have the same number of dimensions.',
        ):

            class BadTransceiver(
                Transceiver, supported_media=[DummyMedium1, DummyMedium2]
            ):
                pass

    def test__init_subclass__with_good_media(self):
        test_media = [MockMedium]

        class GoodTransceiver(Transceiver, supported_media=test_media):
            pass

        self.assertEqual(GoodTransceiver._supported_media, test_media)

    def test__new__(self):
        x1 = self.build_xcvr()
        self.assertEqual(Transceiver._instances[MockTransceiver, x1.name], x1)

        self.assertIs(x1, self.build_xcvr(name=x1.name))

        x2 = self.build_xcvr()

        self.assertEqual(Transceiver._instances[MockTransceiver, x2.name], x2)
        self.assertNotEqual(x1, x2)

    @patch.object(MockTransceiver, 'start')
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    def test__init__(self, PipeMock, mock_start):
        mod_listener = Mock()
        mod_client = Mock()
        osi_listener = Mock()
        osi_client = Mock()

        for auto_start in (True, False):
            PipeMock.side_effect = [
                (mod_listener, mod_client),
                (osi_listener, osi_client),
            ]
            with self.subTest(auto_start=auto_start):

                xcvr = self.build_xcvr(auto_start=auto_start)

                PipeMock.assert_has_calls(
                    [
                        call(duplex=False),
                        call(duplex=False),
                    ]
                )

                self.assertIsNone(xcvr._medium)
                self.assertIsNone(xcvr._tx_queue)
                self.assertIsNone(xcvr._rx_conn)
                self.assertEqual(xcvr._connections_listener, mod_listener)
                self.assertEqual(xcvr._connections_client, mod_client)
                self.assertEqual(xcvr.osi_listener, osi_listener)
                self.assertEqual(xcvr._osi_client, osi_client)

                self.assertIsInstance(xcvr._stop_event, Event)

                if auto_start:
                    mock_start.assert_called_once()
                else:
                    mock_start.assert_not_called()

                self.assertEqual(Transceiver._instances[MockTransceiver, xcvr.name], xcvr)

            mock_start.reset_mock()

            PipeMock.reset_mock()

    def test__init__location(self):

        for dimensionality in (1, 3):
            expected_location = 0.0 if dimensionality == 1 else (0.0, 0.0, 0.0)

            class DummyMedium(Medium, dimensionality=dimensionality):
                pass

            class DummyTransceiver(Transceiver, supported_media=[DummyMedium]):
                def _connect(self, *args, **kwargs):
                    pass

                def _disconnect(self, *args, **kwargs):
                    pass

                def _process_incoming_data(self, *args, **kwargs):
                    pass

                def _process_outgoing_data(self, data, *args, **kwargs):
                    pass

            with self.subTest(dimensionality=dimensionality):
                self.assertEqual(
                    DummyTransceiver(name='test', auto_start=False).location,
                    expected_location,
                )

    def test__repr__(self):
        xcvr = self.build_xcvr()

        self.assertEqual(
            repr(xcvr),
            f'<MockTransceiver: name={xcvr.name!r}, location=disconnected, pid=None>',
        )

        xcvr.start()
        pid = xcvr.pid
        self.assertIsNotNone(pid)
        self.assertEqual(
            repr(xcvr),
            f'<MockTransceiver: name={xcvr.name!r}, location=disconnected, {pid=}>',
        )

        xcvr._medium = medium = self.build_medium()
        xcvr._medium_name.value = medium.name.encode()
        xcvr.location = location = 123.0
        self.assertEqual(
            repr(xcvr),
            (
                f'<MockTransceiver: name={xcvr.name!r}, '
                f'location={medium.name}@{location}, {pid=}>'
            ),
        )

        xcvr.stop()
        self.assertEqual(
            repr(xcvr),
            (
                f'<MockTransceiver: name={xcvr.name!r}, location=disconnected, '
                f'pid={pid}(exited)>'
            ),
        )

    # endregion

    # region Properties

    @patch.object(MockTransceiver, 'connect')
    def test_medium_setter_wraps_connect(self, connect_mock):
        xcvr = self.build_xcvr()
        xcvr.medium = self.medium_mock

        connect_mock.assert_called_once_with(self.medium_mock)

    def test_medium_getter(self):
        xcvr = self.build_xcvr()
        self.assertIsNone(xcvr.medium)

        xcvr._medium = self.medium_mock
        self.assertEqual(xcvr.medium, self.medium_mock)

    def test_location_setter(self):
        xcvr = self.build_xcvr()
        xcvr.location = 123.0
        self.assertEqual(xcvr._location.value, 123.0)

    def test_location_getter(self):
        xcvr = self.build_xcvr()
        self.assertEqual(xcvr._location.value, 0.0)

        xcvr._location.value = 123.0
        self.assertEqual(xcvr.location, 123.0)

    # endregion

    # region connect()

    def test_connect_requires_running_transceiver(self):
        xcvr = self.build_xcvr()
        with self.assertRaisesRegex(
            ProcessNotRunningError,
            'Transceiver must be running before connecting to a Medium',
        ):
            xcvr.connect('foo')

    @patch.object(MockTransceiver, 'disconnect')
    def test_connecting_nothing_same_as_disconnecting(self, disconnect_mock):
        xcvr = self.build_xcvr(is_alive=True, mock_medium=True)

        with self.assertInTargetLogs(
            'DEBUG', 'Connecting to medium=None. Assuming disconnect'
        ):
            xcvr.connect(None)

        disconnect_mock.assert_called_once()

    def test_connecting_same_medium_does_nothing(self):
        xcvr = self.build_xcvr(is_alive=True, mock_medium=True)

        with self.assertNotInTargetLogs(
            'DEBUG',
            'Connecting.*MockTransceiver',
            regex=True,
        ):
            with self.assertInTargetLogs(
                'DEBUG',
                f'{xcvr!r} already connected to {self.medium_mock!r}',
            ):
                xcvr.connect(xcvr._medium)

    def test_connecting_when_already_connected_raises(self):
        xcvr = self.build_xcvr(is_alive=True, mock_medium=True)

        with self.assertRaisesRegex(
            RuntimeError,
            rf'{xcvr!r} already connected to {xcvr._medium!r}. Disconnect first',
        ):
            xcvr.connect(self.medium_mock)

    def test_connect_unsupported_medium_raises(self):
        xcvr = self.build_xcvr(is_alive=True)

        with self.assertRaisesRegex(ValueError, 'Medium.*not supported by.*Transceiver'):
            xcvr.connect(object())

    def test_connect_requires_running_medium(self):
        xcvr = self.build_xcvr(is_alive=True)

        with self.assertRaisesRegex(
            ProcessNotRunningError, 'Transceivers may only connect to running Mediums'
        ):
            xcvr.connect(self.build_medium(is_alive=False))

    def test_connect_bad_send(self):
        xcvr = self.build_xcvr(is_alive=True)
        medium = self.build_medium(mocked=True, is_alive=True)

        err_str = 'foo'
        medium._connection_queue.put.side_effect = OSError(err_str)

        with self.assertRaisesRegex(
            ConnectionError, f'Error sending connection request to {medium!r}: {err_str}'
        ):
            xcvr.connect(medium)

    def test_connect_bad_recv(self):
        xcvr = self.build_xcvr(is_alive=True)
        medium = self.build_medium(mocked=True, is_alive=True)

        err_str = 'bar'

        with patch(f'{BASE_TARGET}.Pipe', autospec=True) as PipeMock:
            PipeMock.return_value = (phy_listener := Mock(), None)

            phy_listener.recv.side_effect = OSError(err_str)

            with self.assertRaisesRegex(
                ConnectionError,
                f'Error receiving connection response from {medium!r}: {err_str}',
            ):
                xcvr.connect(medium)

    def test_connect_bad_response_contents(self):
        xcvr = self.build_xcvr(is_alive=True)
        medium = self.build_medium(mocked=True, is_alive=True)

        with patch(f'{BASE_TARGET}.Pipe', autospec=True) as PipeMock:
            PipeMock.return_value = (phy_listener := Mock(), None)

            phy_listener.recv.return_value = response = 'foo'

            with self.assertRaisesRegex(
                ConnectionError,
                f'Unexpected response contents from {medium!r}: {response!r}',
            ):
                xcvr.connect(medium)

    def test_connect_rejected(self):
        xcvr = self.build_xcvr(is_alive=True)
        medium = self.build_medium(mocked=True, is_alive=True)

        with patch(f'{BASE_TARGET}.Pipe', autospec=True) as PipeMock:
            PipeMock.return_value = (phy_listener := Mock(), None)

            phy_listener.recv.return_value = (
                (result := Responses.ERROR),
                (details := 'foo'),
            )

            with self.assertRaisesRegex(
                ConnectionError,
                f'{medium!r} rejected connection request: {result=}, {details=}',
            ):
                xcvr.connect(medium)

    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch.object(MockTransceiver, '_connect')
    def test_connect_actions(self, connect_mock, PipeMock, *mocks):
        kwargs = {'foo': 'bar'}

        # All of the pipes are generated within the Transeiver. Two pipes in the
        # initializer...
        PipeMock.side_effect = [(None, conn_client_mock := Mock()), (None, None)]

        self.medium_mock._connection_queue = conn_queue = Mock()
        self.medium_mock.is_alive.return_value = True
        xcvr = self.build_xcvr()

        PipeMock.assert_has_calls([call(duplex=False), call(duplex=False)])
        PipeMock.reset_mock()

        # ...and one more in the connect method.
        PipeMock.side_effect = [(phy_listener_mock := Mock(), phy_client_mock := Mock())]

        phy_listener_mock.recv.return_value = (Responses.OK, (location := 123))

        with patch.object(xcvr, 'is_alive', return_value=True):
            with self.assertInTargetLogs(
                'DEBUG', f'Connecting {xcvr!r} to {self.medium_mock!r}'
            ):
                xcvr.connect(self.medium_mock, **kwargs)

        connect_mock.assert_called_once_with(self.medium_mock, **kwargs)
        PipeMock.assert_called_once_with(duplex=False)
        conn_queue.put.assert_called_once_with((phy_client_mock, None, kwargs))
        conn_client_mock.send.assert_called_once_with(phy_listener_mock)
        self.assertEqual(xcvr._medium, self.medium_mock)
        self.assertEqual(xcvr.location, location)

    # endregion

    # region disconnect()

    @patch.object(MockTransceiver, '_disconnect')
    def test_disconnect_called_when_already_disconnected(self, disconnect_mock):
        xcvr = self.build_xcvr()

        with self.assertNotInTargetLogs(
            'DEBUG', 'Disconnecting.*MockTransceiver', regex=True
        ):
            xcvr.disconnect()

        disconnect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch.object(MockTransceiver, '_disconnect')
    def test_disconnect_actions(self, disconnect_mock, PipeMock):
        # All of the pipes are generated within the Transeiver. Two pipes in the
        # initializer...
        PipeMock.side_effect = [(None, conn_client_mock := Mock), (None, None)]
        conn_client_mock.send = Mock()

        self.medium_mock._connection_queue = conn_queue = Mock()

        # Build a pre-connected Transceiver.
        xcvr = self.build_xcvr()
        xcvr.location = location = 123
        xcvr._medium = self.medium_mock

        with self.assertInTargetLogs(
            'DEBUG', f'Disconnecting {xcvr!r} from {self.medium_mock!r}'
        ):
            xcvr.disconnect()

        disconnect_mock.assert_called_once()
        conn_queue.put.assert_called_once_with((None, location, None))
        conn_client_mock.send.assert_called_once_with(None)

        self.assertIsNone(xcvr._medium)

    # endregion

    # region run()-related

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Thread', autospec=True)
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    def test_run_connect_disconnect(self, PipeMock, ThreadMock, EventMock):
        mock_thread = ThreadMock.return_value

        PipeMock.side_effect = [
            (conn_listener_mock := Mock(), None),
            (None, osi_client_mock := Mock()),
        ]
        EventMock.side_effect = [
            proc_stop_event := self.build_unique_mock('is_set'),
            thread_stop_event := Mock(),
        ]

        proc_stop_event.is_set.side_effect = [False, False, True]

        conn_listener_mock.recv.side_effect = [
            phy_listener_mock := self.build_unique_mock('send'),  # connect
            None,  # disconnect
            CLOSE_MSG,  # thread stop
        ]

        xcvr = self.build_xcvr()

        with self.assertInTargetLogs(
            'DEBUG',
            ('Starting transceiver process', f'{xcvr}: shutting down'),
        ):
            xcvr.run()

        self.assertEqual(proc_stop_event.is_set.call_count, 3)

        # Connect
        thread_stop_event.clear.assert_called_once()
        ThreadMock.assert_called_once_with(
            target=xcvr._monitor_medium,
            args=(
                phy_listener_mock,
                osi_client_mock,
                thread_stop_event,
            ),
        )
        mock_thread.start.assert_called_once()

        # Disconnect
        thread_stop_event.set.assert_called_once()
        phy_listener_mock.send.assert_called_once_with(CLOSE_MSG)
        mock_thread.join.assert_called_once()

    @patch.object(MockTransceiver, '_process_incoming_data')
    def test_monitor_medium_bad_recv(self, process_mock):
        xcvr = self.build_xcvr()

        # Prevent cross-talk between unit tests by explicitly setting the methods whose
        # calls we're testing.
        phy_listener_mock = self.build_unique_mock('recv', 'close')
        osi_client_mock = self.build_unique_mock('send')
        stop_event_mock = self.build_unique_mock('is_set')

        stop_event_mock.is_set.side_effect = [False, True]

        phy_listener_mock.recv.side_effect = ValueError(err_str := 'bad recv')

        with self.assertInTargetLogs(
            'ERROR',
            f'{xcvr}: Error receiving data from medium: {err_str}',
        ):
            xcvr._monitor_medium(phy_listener_mock, osi_client_mock, stop_event_mock)

        phy_listener_mock.recv.assert_called_once()
        process_mock.assert_not_called()
        phy_listener_mock.close.assert_called_once()

    @patch.object(MockTransceiver, '_process_incoming_data')
    def test_monitor_medium_bad_data(self, process_mock):
        xcvr = self.build_xcvr()

        # Prevent cross-talk between unit tests by explicitly setting the methods whose
        # calls we're testing.
        phy_listener_mock = self.build_unique_mock('recv', 'close')
        osi_client_mock = self.build_unique_mock('send')
        stop_event_mock = self.build_unique_mock('is_set')

        process_mock.side_effect = ValueError(err_str := 'bad data')

        stop_event_mock.is_set.side_effect = [False, True]

        phy_listener_mock.recv.return_value = data = 0

        with self.assertInTargetLogs(
            'ERROR',
            f'{xcvr}: Error processing {data=}: {err_str}',
        ):
            xcvr._monitor_medium(phy_listener_mock, osi_client_mock, stop_event_mock)

        phy_listener_mock.recv.assert_called_once()
        process_mock.assert_called_once_with(data)
        osi_client_mock.send.assert_not_called()
        self.assertEqual(stop_event_mock.is_set.call_count, 2)
        phy_listener_mock.close.assert_called_once()

    @patch.object(MockTransceiver, '_process_incoming_data')
    def test_monitor_medium_bad_send(self, process_mock):
        xcvr = self.build_xcvr()

        # Prevent cross-talk between unit tests by explicitly setting the methods whose
        # calls we're testing.
        phy_listener_mock = self.build_unique_mock('recv', 'close')
        osi_client_mock = self.build_unique_mock('send')
        stop_event_mock = self.build_unique_mock('is_set')

        process_mock.return_value = out_data = 'blue'

        stop_event_mock.is_set.side_effect = [False, True]

        phy_listener_mock.recv.return_value = in_data = 'red'

        osi_client_mock.send.side_effect = OSError(err_str := 'bad send')

        with self.assertInTargetLogs(
            'ERROR',
            f'{xcvr}: Error sending data={out_data!r} to next layer up: {err_str}',
        ):
            xcvr._monitor_medium(phy_listener_mock, osi_client_mock, stop_event_mock)

        phy_listener_mock.recv.assert_called_once()
        process_mock.assert_called_once_with(in_data)
        osi_client_mock.send.assert_called_once_with(out_data)
        self.assertEqual(stop_event_mock.is_set.call_count, 2)
        phy_listener_mock.close.assert_called_once()

    # endregion

    # region other public methods

    def test_transmit_not_started(self):
        xcvr = self.build_xcvr()

        with self.assertRaisesRegex(
            ProcessNotRunningError,
            'Transceiver must be running before transmitting data',
        ):
            xcvr.transmit(0)

    def test_transmit_no_medium(self):
        xcvr = self.build_xcvr(is_alive=True)

        with self.assertRaisesRegex(
            NoMediumError,
            'Cannot transmit data without a medium',
        ):
            xcvr.transmit(0)

    @patch.object(MockTransceiver, '_process_outgoing_data')
    def test_transmit(self, process_mock):
        xcvr = self.build_xcvr(is_alive=True, mock_medium=True)

        in_data = 'red'
        process_mock.return_value = out_data = 'blue'

        xcvr.transmit(in_data)

        process_mock.assert_called_once_with(in_data)
        xcvr._medium.transmit.assert_called_once_with(out_data, xcvr.location)

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch(f'{BASE_TARGET}.Process.terminate', autospec=True)
    @patch.object(MockTransceiver, 'join')
    def test_stop_in_theory(self, join_mock, terminate_mock, PipeMock, EventMock):
        EventMock.return_value = stop_event_mock = self.build_unique_mock('set')
        PipeMock.side_effect = [
            (None, conn_client_mock := self.build_unique_mock('send')),
            (None, None),
        ]

        xcvr = self.build_xcvr(is_alive=True)
        name = xcvr.name

        xcvr.stop()

        stop_event_mock.set.assert_called_once()
        try:
            conn_client_mock.send.assert_called_once_with(CLOSE_MSG)
        except AssertionError:
            raise

        terminate_mock.assert_called_once_with(xcvr)
        join_mock.assert_called_once()

        self.assertIsNone(Transceiver._instances.get((TestTransceiver, name)))

        # Calling stop() again should not raise an exception.
        xcvr.stop()

    def test_stop_in_practice(self):
        xcvr = self.build_xcvr(auto_start=True)

        self.assertFalse(xcvr._stop_event.is_set())

        xcvr.stop()
        self.assertIsNone(Transceiver._instances.get((TestTransceiver, xcvr.name)))
        self.assertTrue(xcvr._stop_event.is_set())

        # We can still call stop a second time without raising an exception.
        xcvr.stop()

    # endregion


class TestIPC(TestPHYBase):
    def test_send_data_between_transceivers(self):
        # Make sure a broadcast is received by all transceivers.
        names = {'alice', 'bob', 'carla', 'dave', 'eve', 'frank'}
        xvcrs = {n: MockTransceiver(name=n.title()) for n in names}

        medium = MockMedium(name='air')

        for xcvr in xvcrs.values():
            xcvr.connect(medium)

        for sender_name in names:
            receiver_names = names - {sender_name}

            sender = xvcrs[sender_name]
            sender.transmit(tx_data := sender_name[0])

            with self.subTest(sender=sender_name, receivers=receiver_names):
                # Each of the receivers should have received the data.
                for name in receiver_names:
                    receiver = xvcrs[name]

                    if not receiver.osi_listener.poll(timeout=0.1):
                        self.fail(f'No data in {sender_name} to {name} test')
                    self.assertEqual(receiver.osi_listener.recv(), tx_data)

                # The sender should not have received the data.
                self.assertFalse(sender.osi_listener.poll(timeout=0.1))


# endregion

if __name__ == '__main__':
    main()
