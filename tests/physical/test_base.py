#!/usr/bin/env python
from __future__ import annotations

from math import sin, inf as _inf
from multiprocessing import Value
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event, Lock
from unittest import TestCase, main
from unittest.mock import Mock, call, patch

from pynet.physical.base import (
    ConnDetails,
    ReceptionManager,
    Medium,
    Transceiver,
    Transmission,
    _dilate_time,
    _undilate_time,
    cleanup_processes,
)
from pynet.physical.constants import (
    NANOSECONDS_PER_SECOND,
    SPEED_OF_LIGHT,
    TIME_DILATION_FACTOR,
    ManagementMessages,
    Responses,
)
from pynet.physical.exceptions import (
    ConnectionError,
    ProcessNotRunningError,
    StopProcess,
    TransmissionComplete,
)
from pynet.testing import LogTestingMixin, ProcessBuilderMixin

BASE_TARGET = 'pynet.physical.base'

# region Helper Classes


class MockMedium(Medium, dimensionality=1, velocity_factor=0.77):
    """A helper class for testing the :class:`Medium` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _connect(self, *args, **kwargs) -> int:
        with self._conn_details_lock:
            return len(self._conn_details)


class MockTransceiver(Transceiver, supported_media=[MockMedium], buffer_bytes=1500):
    """A helper class for testing the :class:`Transceiver` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _process_rx_amplitude(self, amplitude, *args, **kwargs):
        return amplitude

    def _translate_frame(self, *args, **kwargs):
        return 0


# endregion

# region Tests


class TestHelpers(TestCase):
    def test_dilate_time(self):
        for i in range(3):
            self.assertEqual(_dilate_time(i), i * TIME_DILATION_FACTOR)

    def test_undilate_time(self):
        for i in range(3):
            self.assertEqual(_undilate_time(i * TIME_DILATION_FACTOR), i)


class TestTransmission(TestCase):
    def test_symbol_fn(self):
        with self.subTest('symbol function provided'):
            tx = Transmission(sin, 123, 456, 0.1)
            self.assertEqual(tx._symbol_fn, sin)

        with self.subTest('symbol function not provided'):
            amplitude = 123
            tx = Transmission(amplitude, 123, 456, 0.1)
            self.assertNotEqual(tx._symbol_fn, amplitude)
            self.assertEqual(tx._symbol_fn(0), amplitude)

    def test_get_amplitude(self):
        start_ns = 100
        duration_ns = 200 * TIME_DILATION_FACTOR
        attenuation = 0.5

        tx = Transmission(sin, start_ns, duration_ns, attenuation)

        with self.subTest('pre start'):
            self.assertEqual(tx.get_amplitude(start_ns - 1), 0)

        with self.subTest('start'):
            self.assertEqual(tx.get_amplitude(start_ns), sin(0) * attenuation)

        with self.subTest('middle'):
            middle_ns = duration_ns / 2 + start_ns
            actual_time = (middle_ns - start_ns) / TIME_DILATION_FACTOR

            self.assertEqual(tx.get_amplitude(middle_ns), sin(actual_time) * attenuation)

        with self.subTest('stop'):
            with self.assertRaises(TransmissionComplete):
                tx.get_amplitude(start_ns + duration_ns * TIME_DILATION_FACTOR)


class TestReceptionManager(TestCase, LogTestingMixin):
    log_target: str = BASE_TARGET

    def setUp(self):
        super().setUp()
        self.rx_mgr = ReceptionManager()

    def test_next_rx_ns(self):
        with self.subTest('No transmissions'):
            self.assertEqual(self.rx_mgr.next_rx_ns, _inf)

        self.rx_mgr._transmissions.add(Mock(_start_ns=(start_ns := 123)))
        self.rx_mgr._transmissions.add(Mock(_start_ns=start_ns + 1))

        with self.subTest('Transmissions exist'):
            self.assertEqual(self.rx_mgr.next_rx_ns, start_ns)

    @patch(f'{BASE_TARGET}.Transmission', autospec=True)
    def test_add_transmission(self, TransmissionMock):
        rx_mgr = self.rx_mgr

        symbol_fn = sin
        start_ns = 100
        duration_ns = 200
        attenuation = 0.5

        rx_mgr.add_transmission(symbol_fn, start_ns, duration_ns, attenuation)

        TransmissionMock.assert_called_once_with(
            symbol_fn, start_ns, duration_ns, attenuation
        )
        self.assertEqual(rx_mgr._transmissions, {TransmissionMock.return_value})

    def test_get_amplitude(self):
        rx_mgr = self.rx_mgr

        tx1 = Mock()
        tx2 = Mock()
        tx3 = Mock()
        tx4 = Mock()

        tx3.get_amplitude.side_effect = TransmissionComplete
        tx4.get_amplitude.side_effect = ValueError(err_str := 'test')

        rx_mgr._transmissions = {tx1, tx2, tx3, tx4}

        with self.assertInTargetLogs(
            'ERROR',
            f'{rx_mgr}: an exception occurred while processing a transmission: {err_str}',
        ):
            rx_mgr.get_amplitude(123)

        # The completed transmission was removed from the set.
        self.assertEqual(rx_mgr._transmissions, {tx1, tx2, tx4})


class TestConnDetails(TestCase):
    def test__init__(self):
        loc = 123
        current_ns = 456
        conn_details = ConnDetails(loc, current_ns)

        self.assertEqual(conn_details.loc, loc)
        self.assertEqual(conn_details.last_rx_ns, current_ns)
        self.assertEqual(conn_details.last_tx_ns, current_ns)


class TestPHYBase(TestCase, ProcessBuilderMixin, LogTestingMixin):
    medium_cls: type[Medium] = MockMedium
    xcvr_cls: type[Transceiver] = MockTransceiver
    log_target: str = BASE_TARGET

    def build_unique_mock(self, *names):
        # Prevent cross-talk between unit tests by explicitly setting the attributes
        # under test as brand-new mocks.
        return Mock(**{name: Mock() for name in names})


class TestMedium(TestPHYBase):
    # region Dunders

    def test__init_subclass__(self):
        for dim in (1, 3):
            for vf in (0.1, 1):
                with self.subTest(dimensionality=dim, vf=vf):

                    class DummyMedium(Medium, dimensionality=dim, velocity_factor=vf):
                        pass

                    self.assertEqual(DummyMedium._dimensionality, dim)
                    self.assertEqual(
                        DummyMedium._medium_velocity_ns,
                        vf * SPEED_OF_LIGHT / NANOSECONDS_PER_SECOND,
                    )

    def test__init_subclass__bad_dimensionality(self):
        # Dimensionality cannot be anything other than 1 or 3
        for dimensionality in (0, 2, 4):
            with self.subTest(dimensionality=dimensionality):
                with self.assertRaisesRegex(
                    ValueError, '`dimensionality` must be 1 or 3'
                ):

                    class DummyMedium(
                        Medium, dimensionality=dimensionality, velocity_factor=0.5
                    ):
                        pass

    def test__init_subclass__bad_velocity_factor(self):
        # Velocity factor outside of the range (0, 1] is not allowed.
        for velocity_factor in (-1, 0, 2):
            with self.subTest(velocity_factor=velocity_factor):
                with self.assertRaisesRegex(
                    ValueError, r'`velocity_factor` must be in range \(0, 1\]'
                ):

                    class DummyMedium(
                        Medium, dimensionality=3, velocity_factor=velocity_factor
                    ):
                        pass

    def test__new__(self):
        m1 = self.build_medium()
        self.assertEqual(Medium._instances[MockMedium, m1.name], m1)

        self.assertEqual(m1, self.build_medium(name=m1.name))

        m2 = self.build_medium()

        self.assertEqual(Medium._instances[MockMedium, m2.name], m2)
        self.assertNotEqual(m1, m2)

    @patch.object(MockMedium, '_init_shared_objects')
    @patch.object(MockMedium, 'start')
    def test__init__(self, mock_start, obj_init_mock):
        for auto_start in (True, False):
            with self.subTest(auto_start=auto_start):
                medium = self.build_medium(auto_start=auto_start)

                self.assertEqual(medium._dimensionality, MockMedium._dimensionality)
                self.assertEqual(medium._conn_details, {})

                self.assertIsInstance(medium._conn_details_lock, Lock)
                self.assertIsInstance(medium._connections_client, Connection)
                self.assertIsInstance(medium._connections_listener, Connection)
                self.assertIsInstance(medium._stop_event, Event)

                obj_init_mock.assert_called_once()

                if auto_start:
                    mock_start.assert_called_once()
                else:
                    mock_start.assert_not_called()

                self.assertEqual(Medium._instances[MockMedium, medium.name], medium)

            mock_start.reset_mock()
            obj_init_mock.reset_mock()

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

    # endregion

    # region Multiprocessing- / IPC-related

    def test_connections(self, *mocks):
        medium = self.build_medium()
        medium._conn_details = {1, 2, 3}

        self.assertEqual(medium._connections, tuple(medium._conn_details))

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch(f'{BASE_TARGET}.conn_wait', autospec=True)
    @patch.object(MockMedium, '_connections')
    @patch.object(MockMedium, '_process_connections')
    @patch.object(MockMedium, '_process_tx_event')
    def test_run(
        self,
        process_tx_mock,
        process_connection_mock,
        connections_mock,
        wait_mock,
        PipeMock,
        *mocks,
    ):
        PipeMock.return_value = (conn_listener := Mock(), None)
        medium = self.build_medium()

        def reset_mocks():
            process_tx_mock.reset_mock()
            process_connection_mock.reset_mock()
            connections_mock.reset_mock()
            wait_mock.reset_mock()
            medium._stop_event.is_set.reset_mock()

        with self.subTest('stop right away'):
            medium._stop_event.is_set.side_effect = [True]

            with self.assertInTargetLogs(
                'DEBUG',
                [f'Starting medium process ({medium.pid})', f'{medium}: shutting down'],
            ):
                medium.run()

            wait_mock.assert_not_called()

        reset_mocks()

        with self.subTest('tx event'):
            medium._stop_event.is_set.side_effect = [False, True]

            # Anything other than the connection listener will cause the interpreter to
            # enter the tx event processing block.
            wait_mock.return_value = [rv := 'foo']

            medium.run()

            process_tx_mock.assert_called_once_with(rv)
            process_connection_mock.assert_not_called()

        reset_mocks()

        with self.subTest('new conn'):
            medium._stop_event.is_set.side_effect = [False, True]

            wait_mock.return_value = [conn_listener]

            medium.run()

            process_connection_mock.assert_called_once()
            process_tx_mock.assert_not_called()

        reset_mocks()

        with self.subTest('multiple tx events and new conn'):
            medium._stop_event.is_set.side_effect = [False, True]

            wait_mock.return_value = [rv := 'foo', rv, conn_listener]

            medium.run()

            process_tx_mock.assert_has_calls([call(rv), call(rv)])
            process_connection_mock.assert_called_once()

        reset_mocks()

        with self.subTest('shutting down'):
            medium._stop_event.is_set.side_effect = [False, True]
            process_connection_mock.side_effect = StopProcess

            wait_mock.return_value = [conn_listener]

            medium.run()

            process_connection_mock.assert_called_once()
            medium._stop_event.set.assert_called_once()
            process_tx_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch.object(MockMedium, '_remove_connection')
    @patch.object(MockMedium, '_add_connection')
    def test_process_connections(self, add_mock, remove_mock, PipeMock, *mocks):
        PipeMock.return_value = (conn_listener := Mock(), None)
        conn_poll = conn_listener.poll
        conn_recv = conn_listener.recv

        medium = self.build_medium()

        def reset_mocks():
            conn_poll.reset_mock()
            conn_recv.reset_mock()
            add_mock.reset_mock()
            remove_mock.reset_mock()

            conn_recv.side_effect = None

        with self.subTest('nothing to poll'):
            conn_poll.side_effect = [False]

            medium._process_connections()

            conn_poll.assert_called_once()
            conn_recv.assert_not_called()
            add_mock.assert_not_called()
            remove_mock.assert_not_called()

        reset_mocks()

        with self.subTest('connection closed'):
            conn_poll.side_effect = [True]
            conn_recv.side_effect = EOFError

            with self.assertInTargetLogs(
                'DEBUG', f'{medium}: connection listener closed'
            ):
                with self.assertRaises(StopProcess):
                    medium._process_connections()

            conn_poll.assert_called_once()
            conn_recv.assert_called_once()
            add_mock.assert_not_called()
            remove_mock.assert_not_called()

        reset_mocks()

        with self.subTest('bad connection request'):
            conn_poll.side_effect = [True, False]
            conn_recv.return_value = 'bad request'

            with self.assertInTargetLogs(
                'ERROR', f'{medium}: unexpected connection details type: str'
            ):
                medium._process_connections()

            conn_poll.assert_has_calls([call(), call()])
            conn_recv.assert_called_once()
            add_mock.assert_not_called()
            remove_mock.assert_not_called()

        reset_mocks()

        with self.subTest('add connection'):
            conn_poll.side_effect = [True, False]
            conn_recv.return_value = creq = Mock(create=True)

            medium._process_connections()

            conn_poll.assert_has_calls([call(), call()])
            conn_recv.assert_called_once()
            add_mock.assert_called_once_with(creq)
            remove_mock.assert_not_called()

        reset_mocks()

        with self.subTest('remove connection'):
            conn_poll.side_effect = [True, False]
            conn_recv.return_value = creq = Mock(create=False)

            medium._process_connections()

            conn_poll.assert_has_calls([call(), call()])
            conn_recv.assert_called_once()
            add_mock.assert_not_called()
            remove_mock.assert_called_once_with(creq)

        reset_mocks()

        with self.subTest('add and remove one connection'):
            conn_poll.side_effect = [True, True, False]
            conn_recv.side_effect = [
                creq1 := Mock(create=True),
                creq2 := Mock(create=False),
            ]

            medium._process_connections()

            conn_poll.assert_has_calls([call(), call(), call()])
            conn_recv.assert_has_calls([call(), call()])
            add_mock.assert_called_once_with(creq1)
            remove_mock.assert_called_once_with(creq2)

    @patch(f'{BASE_TARGET}.ConnDetails', autospec=True)
    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    @patch.object(MockMedium, '_connect')
    def test_add_connection(self, connect_mock, mono_mock, ConnDetailsMock):
        medium = self.build_medium()
        mock_conn = Mock()
        conn_req = Mock(conn=mock_conn)
        conn_send = mock_conn.send
        location = 123
        now = mono_mock.return_value

        def reset_mocks():
            connect_mock.reset_mock()
            mono_mock.reset_mock()
            ConnDetailsMock.reset_mock()
            mock_conn.reset_mock()
            conn_send.reset_mock()

            connect_mock.side_effect = None
            conn_send.side_effect = None

        with self.subTest('_connect exception, successful reply'):
            connect_mock.side_effect = err = Exception(err_str := 'connect exception')

            with self.assertInTargetLogs(
                'ERROR',
                f'{medium}: connection failed: {err_str}',
            ):
                medium._add_connection(conn_req)

            connect_mock.assert_called_once_with(conn_req)
            conn_send.assert_called_once_with((Responses.ERROR, err))
            ConnDetailsMock.assert_not_called()

        reset_mocks()

        with self.subTest('_connect exception, reply exception'):
            connect_mock.side_effect = err = Exception(err_str1 := 'connect exception')
            conn_send.side_effect = Exception(err_str2 := 'send exception')

            with self.assertInTargetLogs(
                'ERROR',
                (
                    f'{medium}: connection failed: {err_str1}',
                    f'{medium}: connection error reply to transceiver failed: {err_str2}',
                ),
            ):
                medium._add_connection(conn_req)

            connect_mock.assert_called_once_with(conn_req)
            conn_send.assert_called_once_with((Responses.ERROR, err))
            ConnDetailsMock.assert_not_called()

        reset_mocks()

        with self.subTest('reply error'):
            connect_mock.return_value = location
            conn_send.side_effect = Exception(err_str := 'send exception')

            with self.assertInTargetLogs(
                'ERROR',
                f'{medium}: connection info reply to transceiver failed: {err_str}',
            ):
                medium._add_connection(conn_req)

            connect_mock.assert_called_once_with(conn_req)
            conn_send.assert_called_once_with((Responses.OK, location))
            ConnDetailsMock.assert_called_once_with(location, now)
            self.assertNotIn(location, medium._conn_details)

        reset_mocks()

        with self.subTest('success'):
            connect_mock.return_value = location

            with self.assertInTargetLogs(
                'INFO',
                f'{medium}: connection established at {location=}',
            ):
                medium._add_connection(conn_req)

            connect_mock.assert_called_once_with(conn_req)
            conn_send.assert_called_once_with((Responses.OK, location))
            ConnDetailsMock.assert_called_once_with(location, now)
            self.assertEqual(
                medium._conn_details[mock_conn], ConnDetailsMock.return_value
            )

    @patch.object(Medium, '_disconnect')
    def test_remove_connection(self, disconnect_mock, *mocks):
        medium = self.build_medium()

        location = 123
        conn = Mock()
        creq = Mock(conn=conn, location=location)
        conn_mgr = Mock(loc=location)

        def reset_mocks():
            disconnect_mock.reset_mock()
            conn.reset_mock()

            disconnect_mock.side_effect = None

        with self.subTest('disconnect error'):
            medium._conn_details = {conn: conn_mgr}
            disconnect_mock.side_effect = Exception(err_str := 'disconnect exception')

            with self.assertInTargetLogs(
                'ERROR',
                (
                    f'{medium}: subclass disconnection failed: {err_str}. '
                    'Continuing with removal.'
                ),
            ):
                medium._remove_connection(creq)

            disconnect_mock.assert_called_once_with(creq)
            conn.close.assert_called_once()
            self.assertNotIn(conn, medium._conn_details)

        reset_mocks()

        with self.subTest('success'):
            medium._conn_details = {conn: conn_mgr}

            with self.assertInTargetLogs(
                'INFO',
                f'{medium}: closing connection at {location=}',
            ):
                medium._remove_connection(creq)

            disconnect_mock.assert_called_once_with(creq)
            conn.close.assert_called_once()
            self.assertNotIn(conn, medium._conn_details)

        reset_mocks()

        with self.subTest('missing Connection is ignored'):
            medium._conn_details = {}

            with self.assertInTargetLogs(
                'INFO',
                (
                    f'{medium}: closing connection at {location=}',
                    f'{medium}: connection not found: {conn}',
                ),
            ):
                medium._remove_connection(creq)

            conn.close.assert_not_called()

    @patch.object(Medium, '_effectuate_transmission')
    def test_process_tx_event(self, tx_mock, *mocks):
        medium = self.build_medium()

        src_loc = 123
        src_conn = Mock()
        src_conn_details = Mock(location=src_loc)

        dst_loc1 = 456
        dst_conn1 = Mock()
        dst_conn1_details = Mock(location=dst_loc1)

        dst_loc2 = 789
        dst_conn2 = Mock()
        dst_conn2_details = Mock(location=dst_loc2)

        def reset_mocks():
            tx_mock.reset_mock()
            src_conn.reset_mock()
            dst_conn1.reset_mock()
            dst_conn2.reset_mock()

            src_conn.recv.side_effect = None

            medium._conn_details = {
                src_conn: src_conn_details,
                dst_conn1: dst_conn1_details,
                dst_conn2: dst_conn2_details,
            }

        with self.subTest('closed connection, removed from connections'):
            src_conn.recv.side_effect = EOFError

            with self.assertInTargetLogs(
                'INFO',
                f'{medium}: connection to {src_conn} closed. Removing from connections.',
            ):
                medium._process_tx_event(src_conn)

            tx_mock.assert_not_called()
            self.assertNotIn(src_conn, medium._conn_details)

        reset_mocks()

        with self.subTest('closed connection, not in connections'):
            src_conn.recv.side_effect = EOFError
            medium._conn_details.pop(src_conn)

            with self.assertInTargetLogs(
                'INFO',
                f'{medium}: connection to {src_conn} closed. Removing from connections.',
            ):
                medium._process_tx_event(src_conn)

            tx_mock.assert_not_called()
            self.assertNotIn(src_conn, medium._conn_details)

        reset_mocks()

        with self.subTest('receive error'):
            src_conn.recv.side_effect = Exception(err_str := 'receive exception')

            with self.assertInTargetLogs(
                'ERROR',
                f'{medium}: unexpected exception while receiving tx details: {err_str}',
            ):
                medium._process_tx_event(src_conn)

            tx_mock.assert_not_called()
            self.assertIn(src_conn, medium._conn_details)

        reset_mocks()

        with self.subTest('bad transmission details'):
            src_conn.recv.return_value = bad_details = 'bad transmission details'

            with self.assertInTargetLogs(
                'ERROR',
                f'{medium}: unexpected transmission details: {bad_details}',
            ):
                medium._process_tx_event(src_conn)

            tx_mock.assert_not_called()

        reset_mocks()

        with self.subTest('success for multiple receivers'):
            src_conn.recv.return_value = (
                symbol := 123,
                ns_since_last_xcvr_tx := 456,
                dilated_duration_ns := 789,
            )

            src_conn_details.last_tx_ns = last_tx_ns = 1234

            local_tx_time_ns = last_tx_ns + ns_since_last_xcvr_tx

            medium._process_tx_event(src_conn)

            self.assertEqual(src_conn_details.last_tx_ns, local_tx_time_ns)
            src_conn.recv.assert_called_once()

            tx_mock.assert_has_calls(
                [
                    call(
                        src_loc,
                        dest_conn,
                        dest_conn_details,
                        local_tx_time_ns,
                        dilated_duration_ns,
                        symbol,
                    )
                    for dest_conn, dest_conn_details in medium._conn_details.items()
                    if dest_conn != src_conn
                ]
            )

    @patch.object(Medium, '_calculate_travel_time_ns')
    def test_effectuate_transmission(self, travel_time_mock, *mocks):
        medium = self.build_medium()

        dest_conn = Mock()
        src_loc = 123
        last_rx_ns = 456
        local_tx_time_ns = 1234
        dilated_duration_ns = 5678
        xcvr_rx_delta_ns = local_tx_time_ns - last_rx_ns
        dilated_prop_delay_ns = travel_time_mock.return_value = 9012
        symbol = 9

        with self.subTest('send failed'):
            # Make the source location the first in the call to _calculate_travel_time_ns.
            dest_details = Mock(
                last_rx_ns=last_rx_ns,
                location=(dest_location := src_loc + 1),
            )
            dest_conn.send.side_effect = Exception(err_str := 'send exception')

            with self.assertInTargetLogs(
                'ERROR',
                (
                    f'{medium}: unexpected exception while sending tx details to '
                    f'location={dest_location}: {err_str}'
                ),
            ):
                medium._effectuate_transmission(
                    src_loc,
                    dest_conn,
                    dest_details,
                    local_tx_time_ns,
                    dilated_duration_ns,
                    symbol,
                )

            # Source comes first.
            travel_time_mock.assert_called_once_with(src_loc, dest_location, dilate=True)
            dest_conn.send.assert_called_once_with(
                (
                    symbol,
                    xcvr_rx_delta_ns,
                    dilated_prop_delay_ns,
                    dilated_duration_ns,
                    1,
                )
            )

            # Unchanged due to a failure.
            self.assertEqual(dest_details.last_rx_ns, last_rx_ns)

        travel_time_mock.reset_mock()
        dest_conn.send.reset_mock()
        dest_conn.send.side_effect = None

        with self.subTest('send success'):
            # Make the dest location the first in the call to _calculate_travel_time_ns.
            dest_details = Mock(
                last_rx_ns=last_rx_ns,
                location=(dest_location := src_loc - 1),
            )

            medium._effectuate_transmission(
                src_loc,
                dest_conn,
                dest_details,
                local_tx_time_ns,
                dilated_duration_ns,
                symbol,
            )

            # Destination comes first.
            travel_time_mock.assert_called_once_with(dest_location, src_loc, dilate=True)
            dest_conn.send.assert_called_once_with(
                (
                    symbol,
                    xcvr_rx_delta_ns,
                    dilated_prop_delay_ns,
                    dilated_duration_ns,
                    1,
                )
            )

            # Updated due to a success.
            self.assertEqual(dest_details.last_rx_ns, local_tx_time_ns)

    @patch(f'{BASE_TARGET}._dilate_time', autospec=True)
    @patch(f'{BASE_TARGET}.euclidean_distance', autospec=True)
    def test_calculate_travel_time_ns(self, distance_mock, dilate_mock, *mocks):
        medium = self.build_medium()
        expected_time_ns = 1234
        distance_mock.return_value = expected_time_ns * medium._medium_velocity_ns

        with self.subTest('with dilation'):
            time_ns = medium._calculate_travel_time_ns(
                loc1 := 123, loc2 := 456, dilate=True
            )

            distance_mock.assert_called_once_with(loc1, loc2)
            dilate_mock.assert_called_once_with(expected_time_ns)

            self.assertEqual(time_ns, dilate_mock.return_value)

        distance_mock.reset_mock()
        dilate_mock.reset_mock()

        with self.subTest('without dilation'):
            time_ns = medium._calculate_travel_time_ns(
                loc1 := 123, loc2 := 456, dilate=False
            )

            distance_mock.assert_called_once_with(loc1, loc2)
            dilate_mock.assert_not_called()

            self.assertEqual(time_ns, expected_time_ns)

    # endregion

    # region Public Methods

    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch(f'{BASE_TARGET}.ConnRequest', autospec=True)
    def test_connect(self, ConnReqMock, PipeMock, *mocks):
        PipeMock.return_value = None, (client := Mock())
        medium = self.build_medium()
        conn = Mock()
        location = 123
        kwargs = {'foo': 'bar'}

        medium.connect(conn, location, **kwargs)

        ConnReqMock.assert_called_once_with(conn, location, True, kwargs)
        client.send.assert_called_once_with(ConnReqMock.return_value)

    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch(f'{BASE_TARGET}.ConnRequest', autospec=True)
    def test_disconnect(self, ConnReqMock, PipeMock, *mocks):
        PipeMock.return_value = None, (client := Mock())
        medium = self.build_medium()
        conn = Mock()
        location = 123
        kwargs = {'foo': 'bar'}

        medium.disconnect(conn, location, **kwargs)

        ConnReqMock.assert_called_once_with(conn, location, False, kwargs)
        client.send.assert_called_once_with(ConnReqMock.return_value)

    @patch(f'{BASE_TARGET}.Process.terminate', autospec=True)
    @patch.object(MockMedium, 'join')
    def test_stop_in_theory(self, join_mock, terminate_mock):
        medium = self.build_medium()
        medium._stop_event = stop_event = Mock()
        medium._connections_client = conn_client = Mock()
        medium.is_alive = Mock(return_value=True)

        with self.subTest('Queue is closed'):
            conn_client.send.side_effect = ValueError
            medium.stop()

            stop_event.set.assert_called_once()
            conn_client.send.assert_called_once_with(ManagementMessages.CLOSE)

            join_mock.assert_called_once()
            terminate_mock.assert_called_once_with(medium)

            self.assertIsNone(Medium._instances.get((TestMedium, medium.name)))

            medium.stop()

        # Build another medium for the second subtest
        medium = self.build_medium()
        medium._stop_event = stop_event = Mock()
        medium._connections_client = conn_client = Mock()
        medium.is_alive = Mock(return_value=True)
        join_mock.reset_mock()
        terminate_mock.reset_mock()

        with self.subTest('Queue is still open'):
            medium.stop()

            stop_event.set.assert_called_once()
            conn_client.send.assert_called_once_with(ManagementMessages.CLOSE)

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

    # endregion

    # region misc

    def test_init_shared_objects(self):
        test_val = 123

        class MyMedium(MockMedium, dimensionality=1, velocity_factor=0.77):
            def _init_shared_objects(self):
                self._test_val = Value('i', test_val)

        medium = MyMedium(name='foo')
        self.assertEqual(medium._test_val.value, test_val)

    # endregion


class TestTransceiver(TestPHYBase):

    # region Dunders

    def test__init_subclass__(self):
        with self.subTest('with non-medium media'):
            with self.assertRaisesRegex(TypeError, 'are not subclasses of Medium'):

                class BadTransceiver1(
                    Transceiver, supported_media=[object], buffer_bytes=10
                ):
                    pass

        # Valid dimensionalities, but they differ and that's not allowed.
        class DummyMedium1(Medium, dimensionality=1, velocity_factor=1):
            pass

        class DummyMedium2(Medium, dimensionality=3, velocity_factor=1):
            pass

        with self.subTest('with media of different dimensionalities'):
            with self.assertRaisesRegex(
                ValueError,
                '`supported_media` must have the same number of dimensions.',
            ):

                class BadTransceiver2(
                    Transceiver,
                    supported_media=[DummyMedium1, DummyMedium2],
                    buffer_bytes=10,
                ):
                    pass

        with self.subTest('with non-positive integer buffer_bytes'):
            with self.assertRaisesRegex(
                ValueError, '`buffer_bytes` must be a positive integer.'
            ):

                class BadTransceiver3(
                    Transceiver, supported_media=[MockMedium], buffer_bytes=0
                ):
                    pass

        test_media = [MockMedium]
        buffer_bytes = 20

        with self.subTest('with valid arguments'):

            class GoodTransceiver(
                Transceiver, supported_media=test_media, buffer_bytes=buffer_bytes
            ):
                pass

            self.assertEqual(GoodTransceiver._supported_media, test_media)
            self.assertEqual(GoodTransceiver._buffer_bytes, buffer_bytes)

    def test__new__(self):
        x1 = self.build_xcvr()
        self.assertEqual(Transceiver._instances[MockTransceiver, x1.name], x1)

        self.assertIs(x1, self.build_xcvr(name=x1.name))

        x2 = self.build_xcvr()

        self.assertEqual(Transceiver._instances[MockTransceiver, x2.name], x2)
        self.assertNotEqual(x1, x2)

    @patch.object(MockTransceiver, 'start')
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch(f'{BASE_TARGET}._dilate_time', autospec=True)
    def test__init__(self, dilate_mock, PipeMock, mock_start):
        conn_listener = Mock()
        conn_client = Mock()
        l2_osi_conn = Mock()
        l1_osi_conn = Mock()

        base_baud = 1e6

        for auto_start in (True, False):
            PipeMock.side_effect = [
                (conn_listener, conn_client),
                (l2_osi_conn, l1_osi_conn),
            ]
            with self.subTest(auto_start=auto_start):

                xcvr = self.build_xcvr(auto_start=auto_start, base_baud=base_baud)

                PipeMock.assert_has_calls(
                    [
                        call(duplex=False),  # Connections
                        call(duplex=True),  # OSI
                    ]
                )

                self.assertIsNone(xcvr._medium)

                self.assertIsNone(xcvr._next_rx_ns)
                self.assertIsNone(xcvr._next_tx_ns)
                self.assertIsNone(xcvr._last_medium_rx_ns)
                self.assertIsNone(xcvr._last_medium_tx_ns)

                self.assertEqual(xcvr._tx_frame_symbol_len, 0)
                self.assertEqual(xcvr._tx_index, 0)
                self.assertIsNone(xcvr._tx_frame_symbols)

                self.assertEqual(xcvr._base_delta_ns, NANOSECONDS_PER_SECOND // base_baud)
                self.assertEqual(xcvr._dilated_base_delta_ns, dilate_mock.return_value)
                dilate_mock.assert_called_once_with(xcvr._base_delta_ns)

                self.assertEqual(xcvr._connections_listener, conn_listener)
                self.assertEqual(xcvr._connections_client, conn_client)
                self.assertEqual(xcvr.l2_osi_conn, l2_osi_conn)
                self.assertEqual(xcvr._l1_osi_conn, l1_osi_conn)

                self.assertIsNone(xcvr._conn_2_med)
                self.assertIsNone(xcvr._conn_in_medium)

                self.assertIsInstance(xcvr._rx_manager, ReceptionManager)
                self.assertIsInstance(xcvr._rx_manager_lock, Lock)

                self.assertIsInstance(xcvr._stop_event, Event)

                if auto_start:
                    mock_start.assert_called_once()
                else:
                    mock_start.assert_not_called()

                self.assertEqual(Transceiver._instances[MockTransceiver, xcvr.name], xcvr)

            mock_start.reset_mock()
            dilate_mock.reset_mock()
            PipeMock.reset_mock()

    def test__init__location(self):

        for dimensionality in (1, 3):
            expected_location = 0.0 if dimensionality == 1 else (0.0, 0.0, 0.0)

            class DummyMedium(Medium, dimensionality=dimensionality, velocity_factor=1):
                pass

            class DummyTransceiver(
                Transceiver, supported_media=[DummyMedium], buffer_bytes=10
            ):
                def _connect(self, *args, **kwargs):
                    pass

                def _disconnect(self, *args, **kwargs):
                    pass

                def _process_rx_amplitude(self, symbol):
                    pass

                def _translate_frame(self):
                    pass

            with self.subTest(dimensionality=dimensionality):
                self.assertEqual(
                    DummyTransceiver(
                        name='test', auto_start=False, base_baud=1e6
                    ).location,
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

    def test_location_getter(self):
        with self.subTest('1-dimensional'):
            xcvr = self.build_xcvr()
            self.assertEqual(xcvr.location, 0.0)

        with self.subTest('3-dimensional'):

            class DummyMedium(Medium, dimensionality=3, velocity_factor=1):
                pass

            class DummyTransceiver(
                Transceiver, supported_media=[DummyMedium], buffer_bytes=10
            ):
                def _connect(self, *args, **kwargs):
                    pass

                def _disconnect(self, *args, **kwargs):
                    pass

                def _process_rx_amplitude(self, symbol):
                    pass

                def _translate_frame(self):
                    pass

            xcvr = DummyTransceiver(name='test', auto_start=False, base_baud=1e6)
            self.assertEqual(xcvr.location, (0.0, 0.0, 0.0))

    def test_location_setter(self):
        with self.subTest('1-dimensional, good input'):
            xcvr = self.build_xcvr()
            xcvr.location = 123.0
            self.assertEqual(xcvr._location[:], [123.0])

        with self.subTest('1-dimensional, bad input'):
            xcvr = self.build_xcvr()
            with self.assertRaisesRegex(
                ValueError,
                'Expected a sequence of length 1, got sequence of length 2 instead',
            ):
                xcvr.location = (123.0, 3.4)

        with self.subTest('3-dimensional, good input'):

            class DummyMedium(Medium, dimensionality=3, velocity_factor=1):
                pass

            class DummyTransceiverLoc1(
                Transceiver, supported_media=[DummyMedium], buffer_bytes=10
            ):
                def _connect(self, *args, **kwargs):
                    pass

                def _disconnect(self, *args, **kwargs):
                    pass

                def _process_rx_amplitude(self, symbol):
                    pass

                def _translate_frame(self):
                    pass

            xcvr = DummyTransceiverLoc1(name='test', auto_start=False, base_baud=1e6)
            xcvr.location = (1.0, 2.0, 3.0)
            self.assertEqual(xcvr._location[:], [1.0, 2.0, 3.0])

        with self.subTest('3-dimensional, bad input'):

            class DummyMedium(Medium, dimensionality=3, velocity_factor=1):
                pass

            class DummyTransceiverLoc2(
                Transceiver, supported_media=[DummyMedium], buffer_bytes=10
            ):
                def _connect(self, *args, **kwargs):
                    pass

                def _disconnect(self, *args, **kwargs):
                    pass

                def _process_rx_amplitude(self, symbol):
                    pass

                def _translate_frame(self):
                    pass

            xcvr = DummyTransceiverLoc2(name='test', auto_start=False, base_baud=1e6)
            with self.assertRaisesRegex(
                ValueError,
                'Expected a sequence of length 3, got a scalar instead',
            ):
                xcvr.location = 1

        with self.subTest('bad input type'):
            xcvr = self.build_xcvr()
            with self.assertRaisesRegex(
                TypeError, 'Expected a sequence or scalar for location, got str instead'
            ):
                xcvr.location = 'test'

    # endregion

    # region run()-related

    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Thread', autospec=True)
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    def test_run(self, PipeMock, ThreadMock, EventMock, mono_mock):
        mock_thread = ThreadMock.return_value

        with self.subTest('close immediately'):
            PipeMock.side_effect = [
                (conn_listener_mock := Mock(), None),
                (None, None),
            ]
            EventMock.side_effect = [
                proc_stop_event := self.build_unique_mock('is_set'),
            ]

            proc_stop_event.is_set.side_effect = [True]

            xcvr = self.build_xcvr()
            xcvr.run()

            proc_stop_event.is_set.assert_called_once()
            conn_listener_mock.recv.assert_not_called()

        with self.subTest('connect, disconnect, close'):
            PipeMock.side_effect = [
                (conn_listener_mock := Mock(), None),
                (None, None),
            ]
            EventMock.side_effect = [
                proc_stop_event := self.build_unique_mock('is_set'),
            ]

            proc_stop_event.is_set.side_effect = [False, False, False]

            conn_listener_mock.recv.side_effect = [
                conn := self.build_unique_mock('send'),  # connect
                None,  # disconnect
                ManagementMessages.CLOSE,  # thread stop
            ]

            xcvr = self.build_xcvr()

            with self.assertInTargetLogs(
                'DEBUG',
                ('Starting transceiver process', f'{xcvr}: shutting down'),
            ):
                xcvr.run()

            self.assertEqual(proc_stop_event.is_set.call_count, 3)

            # Connect
            mono_mock.assert_called_once()
            ThreadMock.assert_called_once_with(target=xcvr._monitor_medium)
            mock_thread.start.assert_called_once()

            # Disconnect
            conn.send.assert_called_once_with(ManagementMessages.CLOSE)
            conn.close.assert_called_once()
            mock_thread.join.assert_called_once()
            self.assertIsNone(xcvr._last_medium_tx_ns)
            self.assertIsNone(xcvr._conn_2_med)
            self.assertIsNone(xcvr._medium)

        mono_mock.reset_mock()
        ThreadMock.reset_mock()

        with self.subTest('connect, close before disconnect'):
            PipeMock.side_effect = [
                (conn_listener_mock := Mock(), None),
                (None, None),
            ]
            EventMock.side_effect = [
                proc_stop_event := self.build_unique_mock('is_set'),
            ]

            proc_stop_event.is_set.side_effect = [False, False]

            conn_listener_mock.recv.side_effect = [
                conn := self.build_unique_mock('send'),  # connect
                ManagementMessages.CLOSE,  # thread stop
            ]

            xcvr = self.build_xcvr()

            with self.assertInTargetLogs(
                'DEBUG',
                ('Starting transceiver process', f'{xcvr}: shutting down'),
            ):
                xcvr.run()

            self.assertEqual(proc_stop_event.is_set.call_count, 2)

            # Connect
            mono_mock.assert_called_once()
            ThreadMock.assert_called_once_with(target=xcvr._monitor_medium)
            mock_thread.start.assert_called_once()

            # Close
            conn.send.assert_called_once_with(ManagementMessages.CLOSE)
            mock_thread.join.assert_called_once()

            # We don't bother resetting values because we're shutting down.
            self.assertEqual(xcvr._last_medium_tx_ns, mono_mock.return_value)

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.conn_wait', autospec=True)
    @patch.object(MockTransceiver, '_process_receptions')
    @patch.object(MockTransceiver, '_process_transmissions')
    @patch.object(MockTransceiver, '_process_new_medium_reception')
    @patch.object(MockTransceiver, '_process_new_l2_frame')
    def test_monitor_medium(
        self,
        new_l2_rx_mock,
        new_med_rx_mock,
        process_known_tx_mock,
        process_known_rx_mock,
        mono_mock,
        wait_mock,
        EventMock,
    ):
        stop_event_mock = EventMock.return_value
        xcvr = self.build_xcvr()

        xcvr._conn_2_med = med_conn = Mock()
        xcvr._l1_osi_conn = l1_conn = Mock()

        with self.subTest('no known events + medium reception'):
            wait_mock.return_value = [med_conn]
            stop_event_mock.is_set.side_effect = [False, True]

            xcvr._monitor_medium()

            process_known_rx_mock.assert_called_once()
            process_known_tx_mock.assert_called_once()

            wait_mock.assert_called_once_with((med_conn, l1_conn), timeout=None)

            new_med_rx_mock.assert_called_once()
            new_l2_rx_mock.assert_not_called()

            stop_event_mock.is_set.assert_has_calls([call(), call()])

        wait_mock.reset_mock()
        stop_event_mock.is_set.reset_mock()
        process_known_rx_mock.reset_mock()
        process_known_tx_mock.reset_mock()
        new_l2_rx_mock.reset_mock()
        new_med_rx_mock.reset_mock()

        with self.subTest('next_tx earlier than next_rx + L2 reception'):
            xcvr._next_tx_ns = next_tx_ns = 1
            xcvr._next_rx_ns = next_tx_ns + 1
            mono_mock.return_value = 0
            wait_mock.return_value = [l1_conn]
            stop_event_mock.is_set.side_effect = [False, True]

            xcvr._monitor_medium()

            process_known_rx_mock.assert_called_once()
            process_known_tx_mock.assert_called_once()

            wait_mock.assert_called_once_with((med_conn, l1_conn), timeout=next_tx_ns)

            new_med_rx_mock.assert_not_called()
            new_l2_rx_mock.assert_called_once()

            stop_event_mock.is_set.assert_has_calls([call(), call()])

    @patch(f'{BASE_TARGET}._dilate_time', autospec=True)
    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    @patch.object(MockTransceiver, '_process_rx_amplitude')
    def test_process_receptions(self, process_mock, mono_mock, dilate_mock):
        xcvr = self.build_xcvr()
        xcvr._rx_manager = rx_manager = self.build_unique_mock('get_amplitude')
        get_amplitude = rx_manager.get_amplitude
        mono_mock.return_value = current_time = 100
        amplitude = 1.3

        # _dilate_time was called during the constructor, so we reset it here.
        dilate_mock.reset_mock()

        def reset_mocks():
            process_mock.reset_mock()
            get_amplitude.reset_mock()
            mono_mock.reset_mock()
            dilate_mock.reset_mock()

            mono_mock.return_value = current_time

        with self.subTest('no known rx events'):
            xcvr._next_rx_ns = None
            rx_manager.next_rx_ns = None

            xcvr._process_receptions()

            mono_mock.assert_called_once()
            process_mock.assert_not_called()
            get_amplitude.assert_not_called()
            dilate_mock.assert_not_called()

        reset_mocks()

        with self.subTest('not yet time to process known rx events'):
            xcvr._next_rx_ns = current_time + 1
            rx_manager.next_rx_ns = current_time + 1

            xcvr._process_receptions()

            mono_mock.assert_called_once()
            process_mock.assert_not_called()
            get_amplitude.assert_not_called()
            dilate_mock.assert_not_called()

        reset_mocks()

        with self.subTest('time to process last rx event in transmission'):
            xcvr._next_rx_ns = current_time
            get_amplitude.return_value = amplitude
            process_mock.return_value = -1

            xcvr._process_receptions()

            mono_mock.assert_called_once()
            get_amplitude.assert_called_once_with(current_time)
            process_mock.assert_called_once_with(amplitude)
            dilate_mock.assert_not_called()
            self.assertIsNone(xcvr._next_rx_ns)

        reset_mocks()

        with self.subTest('time to process rx event in transmission, delta not given'):
            dilate_mock.return_value = dilated_time = 1000
            rx_manager.next_rx_ns = current_time
            get_amplitude.return_value = amplitude
            process_mock.return_value = None

            xcvr._process_receptions()

            mono_mock.assert_called_once()
            get_amplitude.assert_called_once_with(current_time)
            process_mock.assert_called_once_with(amplitude)
            dilate_mock.assert_called_once_with(xcvr._base_delta_ns / 2)

            self.assertEqual(xcvr._next_rx_ns, current_time + dilated_time)

        reset_mocks()

        with self.subTest('time to process rx event in transmission, delta given'):
            dilate_mock.return_value = dilated_time = 1000
            xcvr._next_rx_ns = current_time
            rx_manager.next_rx_ns = current_time
            get_amplitude.return_value = amplitude
            process_mock.return_value = undilated_time = 123

            xcvr._process_receptions()

            mono_mock.assert_called_once()
            get_amplitude.assert_called_once_with(current_time)
            process_mock.assert_called_once_with(amplitude)
            dilate_mock.assert_called_once_with(undilated_time)

            self.assertEqual(xcvr._next_rx_ns, current_time + dilated_time)

    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    def test_process_transmissions(self, mono_mock):
        xcvr = self.build_xcvr()
        xcvr._conn_2_med = conn = self.build_unique_mock('send')
        send = conn.send
        mono_mock.return_value = current_time = 100

        def reset_mocks():
            mono_mock.reset_mock()
            send.reset_mock()

            send.side_effect = None

        with self.subTest('no transmission'):
            xcvr._next_tx_ns = None

            xcvr._process_transmissions()

            mono_mock.assert_not_called()
            send.assert_not_called()

        reset_mocks()

        with self.subTest('not yet time to transmit'):
            xcvr._next_tx_ns = current_time + 1

            xcvr._process_transmissions()

            send.assert_not_called()
            mono_mock.assert_called_once()

        reset_mocks()

        with self.subTest('just sent last symbol'):
            xcvr._tx_index = 1
            xcvr._tx_frame_symbol_len = 1
            xcvr._next_tx_ns = current_time
            xcvr._tx_frame_symbols = [1]

            xcvr._process_transmissions()

            send.assert_not_called()
            mono_mock.assert_called_once()

            self.assertIsNone(xcvr._tx_frame_symbols)
            self.assertIsNone(xcvr._tx_frame_symbol_len)
            self.assertEqual(xcvr._tx_index, 0)
            self.assertIsNone(xcvr._next_tx_ns)

        reset_mocks()

        with self.subTest('error sending next symbol'):
            xcvr._tx_index = 1
            xcvr._tx_frame_symbol_len = 2
            xcvr._next_tx_ns = current_time
            xcvr._last_medium_tx_ns = current_time - 1
            xcvr._dilated_base_delta_ns = dilated_duration_ns = 1000
            xcvr._tx_frame_symbols = [1, (symbol := 2)]
            send.side_effect = Exception(err_str := 'I am an error')
            new_next_tx_ns = current_time + dilated_duration_ns

            with self.assertInTargetLogs(
                'ERROR', f'{xcvr}: Error sending symbol: {err_str}'
            ):
                xcvr._process_transmissions()

            send.assert_called_once_with((symbol, 1, dilated_duration_ns))
            mono_mock.assert_called_once()

            self.assertEqual(xcvr._tx_index, 2)
            self.assertEqual(xcvr._last_medium_tx_ns, current_time)
            self.assertEqual(xcvr._next_tx_ns, new_next_tx_ns)

        reset_mocks()

        with self.subTest('successfuly sent next symbol'):
            xcvr._tx_index = 1
            xcvr._tx_frame_symbol_len = 2
            xcvr._next_tx_ns = current_time
            xcvr._last_medium_tx_ns = current_time - 1
            xcvr._dilated_base_delta_ns = dilated_duration_ns = 1000
            xcvr._tx_frame_symbols = [1, (symbol := 2)]
            new_next_tx_ns = current_time + dilated_duration_ns

            with self.assertNotInTargetLogs(
                'ERROR', f'{xcvr}: Error sending symbol: {err_str}'
            ):
                xcvr._process_transmissions()

            send.assert_called_once_with((symbol, 1, dilated_duration_ns))
            mono_mock.assert_called_once()

            self.assertEqual(xcvr._tx_index, 2)
            self.assertEqual(xcvr._last_medium_tx_ns, current_time)
            self.assertEqual(xcvr._next_tx_ns, new_next_tx_ns)

    def test_process_new_medium_reception(self):
        xcvr = self.build_xcvr()
        xcvr._rx_manager = rx_manager = self.build_unique_mock('add_transmission')
        add_transmission = rx_manager.add_transmission
        xcvr._conn_2_med = conn = self.build_unique_mock('recv')
        recv = conn.recv

        def reset_mocks():
            add_transmission.reset_mock()
            recv.reset_mock()

            recv.side_effect = None

        with self.subTest('bad recv'):
            recv.side_effect = Exception

            with self.assertInTargetLogs(
                'ERROR', f'{xcvr}: Error receiving symbol details from medium'
            ):
                xcvr._process_new_medium_reception()

            recv.assert_called_once()
            add_transmission.assert_not_called()

        reset_mocks()

        with self.subTest('bad symbol details'):
            recv.return_value = symbol_details = 'I am not a tuple'

            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Received invalid symbol details from medium: {symbol_details!r}',
            ):
                xcvr._process_new_medium_reception()

            recv.assert_called_once()
            add_transmission.assert_not_called()

        reset_mocks()

        with self.subTest('good symbol'):
            symbol = 1
            delta_ns = 2
            dilated_prop_delay_ns = 3
            dilated_duration_ns = 4
            attenuation = 5
            recv.return_value = symbol_details = (
                symbol,
                delta_ns,
                dilated_prop_delay_ns,
                dilated_duration_ns,
                attenuation,
            )

            xcvr._last_medium_rx_ns = last_time = 1000
            local_rx_time_ns = last_time + delta_ns

            xcvr._process_new_medium_reception()

            add_transmission.assert_called_once_with(
                symbol,
                local_rx_time_ns + dilated_prop_delay_ns,
                dilated_duration_ns,
                attenuation,
            )
            recv.assert_called_once()
            self.assertEqual(xcvr._last_medium_rx_ns, local_rx_time_ns)

    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    @patch.object(MockTransceiver, '_translate_frame')
    def test_process_new_l2_frame(self, translate_mock, mono_mock):
        xcvr = self.build_xcvr()
        xcvr._l1_osi_conn = l1_osi_conn = self.build_unique_mock('recv')
        recv = l1_osi_conn.recv

        def reset_mocks():
            translate_mock.reset_mock()
            mono_mock.reset_mock()
            recv.reset_mock()

            translate_mock.side_effect = None
            recv.side_effect = None

        with self.subTest('bad recv'):
            recv.side_effect = Exception

            with self.assertInTargetLogs(
                'ERROR', f'{xcvr}: Error receiving frame from next layer up'
            ):
                xcvr._process_new_l2_frame()

            recv.assert_called_once()
            translate_mock.assert_not_called()

        reset_mocks()

        with self.subTest('failed translation'):
            recv.return_value = frame = 'I am a frame'
            translate_mock.side_effect = Exception(err_str := 'I am an error')

            with self.assertInTargetLogs(
                'ERROR', f'{xcvr}: Error translating frame: {err_str}'
            ):
                xcvr._process_new_l2_frame()

            recv.assert_called_once()
            translate_mock.assert_called_once_with(frame)
            mono_mock.assert_not_called()

        reset_mocks()

        with self.subTest('successful translation'):
            recv.return_value = frame = 'I am a frame'
            translate_mock.return_value = translated_frame = [1, 2, 3]
            mono_mock.return_value = current_time = 1000

            xcvr._process_new_l2_frame()

            recv.assert_called_once()
            translate_mock.assert_called_once_with(frame)
            mono_mock.assert_called_once()
            self.assertEqual(xcvr._next_tx_ns, current_time)
            self.assertEqual(xcvr._tx_index, 0)
            self.assertEqual(xcvr._tx_frame_symbols, translated_frame)

    # endregion

    # region public methods

    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch.object(MockTransceiver, 'disconnect')
    @patch.object(MockTransceiver, '_connect')
    def test_connect(self, connect_mock, disconnect_mock, PipeMock):
        medium_conn = Mock()
        xcvr_conn = Mock()
        PipeMock.return_value = (Mock(), Mock())

        xcvr = self.build_xcvr()
        medium = self.build_medium(mocked=True)

        def reset_mocks():
            medium.connect.reset_mock()
            disconnect_mock.reset_mock()
            connect_mock.reset_mock()
            PipeMock.reset_mock()
            xcvr_conn.reset_mock()

            connect_mock.side_effect = None
            xcvr_conn.recv.side_effect = None
            medium.connect.side_effect = None
            PipeMock.return_value = (medium_conn, xcvr_conn)

        with self.subTest('transceiver not running'):
            with self.assertRaisesRegex(
                ProcessNotRunningError,
                'Transceiver must be running before connecting to a Medium',
            ):
                xcvr.connect('foo')

        reset_mocks()
        xcvr = self.build_xcvr(is_alive=True)

        with self.subTest('connecting nothing same as disconnecting'):
            xcvr._medium = medium

            with self.assertInTargetLogs(
                'DEBUG', 'Connecting to medium=None. Assuming disconnect'
            ):
                xcvr.connect(None)

            disconnect_mock.assert_called_once()

        reset_mocks()

        with self.subTest('connecting same medium does nothing'):
            with self.assertNotInTargetLogs(
                'DEBUG',
                'Connecting.*MockTransceiver',
                regex=True,
            ):
                with self.assertInTargetLogs(
                    'DEBUG',
                    f'{xcvr!r} already connected to {medium!r}',
                ):
                    xcvr.connect(medium)

        reset_mocks()

        with self.subTest('connecting when already connected raises'):
            with self.assertRaisesRegex(
                RuntimeError,
                rf'{xcvr!r} already connected to {xcvr._medium!r}. Disconnect first',
            ):
                xcvr.connect('different medium')

        reset_mocks()
        xcvr._medium = None

        with self.subTest('connect_unsupported_medium_raises'):
            with self.assertRaisesRegex(
                ValueError, 'Medium.*not supported by.*Transceiver'
            ):
                xcvr.connect(object())

        reset_mocks()

        with self.subTest('connect requires running medium'):
            with self.assertRaisesRegex(
                ProcessNotRunningError, 'Transceivers may only connect to running Mediums'
            ):
                xcvr.connect(self.build_medium(is_alive=False))

        reset_mocks()

        with self.subTest('connect bad recv'):
            err_str = 'bar'

            xcvr_conn.recv.side_effect = OSError(err_str)

            with self.assertRaisesRegex(
                ConnectionError,
                f'Error receiving connection response from {medium!r}: {err_str}',
            ):
                xcvr.connect(medium)

            xcvr_conn.recv.assert_called_once()

        reset_mocks()

        with self.subTest('connect bad response contents'):
            xcvr_conn.recv.return_value = response = 'foo'

            with self.assertRaisesRegex(
                ConnectionError,
                f'Unexpected response contents from {medium!r}: {response!r}',
            ):
                xcvr.connect(medium)

        reset_mocks()

        with self.subTest('connect rejected'):
            xcvr_conn.recv.return_value = (
                (result := Responses.ERROR),
                (details := 'foo'),
            )

            with self.assertRaisesRegex(
                ConnectionError,
                f'{medium!r} rejected connection request: {result=}, {details=}',
            ):
                xcvr.connect(medium)

        medium = self.build_medium(mocked=True, is_alive=True)
        reset_mocks()

        with self.subTest('connect bad send'):
            err_str = 'foo'
            medium.connect.side_effect = OSError(err_str)

            with self.assertRaisesRegex(
                ConnectionError,
                f'Error sending connection request to {medium!r}: {err_str}',
            ):
                xcvr.connect(medium)

            medium.connect.assert_called_once_with(medium_conn)
            PipeMock.assert_called_once_with(duplex=True)

        reset_mocks()
        xcvr._connections_client = conn_client = Mock()

        with self.subTest('full connect actions'):
            kwargs = {'foo': 'bar'}

            xcvr_conn.recv.return_value = (Responses.OK, (location := 123))

            with self.assertInTargetLogs('DEBUG', f'Connecting {xcvr!r} to {medium!r}'):
                xcvr.connect(medium, **kwargs)

            connect_mock.assert_called_once_with(medium, **kwargs)
            PipeMock.assert_called_once_with(duplex=True)
            medium.connect.assert_called_once_with(medium_conn, **kwargs)
            conn_client.send.assert_called_once_with(xcvr_conn)
            self.assertEqual(xcvr._medium, medium)
            self.assertEqual(xcvr.location, location)

    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch.object(MockTransceiver, '_disconnect')
    def test_disconnect(self, disconnect_mock, PipeMock):
        medium_conn = Mock()
        xcvr_conn = Mock()
        PipeMock.return_value = (Mock(), Mock())

        xcvr = self.build_xcvr()
        medium = self.build_medium(mocked=True)

        def reset_mocks():
            medium.disconnect.reset_mock()
            disconnect_mock.reset_mock()
            PipeMock.reset_mock()
            xcvr_conn.reset_mock()

            xcvr_conn.recv.side_effect = None
            medium.disconnect.side_effect = None
            PipeMock.return_value = (medium_conn, xcvr_conn)

        with self.subTest('disconnect called when already disconnected'):

            with self.assertNotInTargetLogs(
                'DEBUG', 'Disconnecting.*MockTransceiver', regex=True
            ):
                xcvr.disconnect()

            disconnect_mock.assert_not_called()

        reset_mocks()
        xcvr._connections_client = conn_client = Mock()

        with self.subTest('disconnect actions'):
            xcvr.location = 123
            xcvr._medium = medium
            xcvr._conn_in_medium = medium_conn = Mock()

            with self.assertInTargetLogs(
                'DEBUG', f'Disconnecting {xcvr!r} from {medium!r}'
            ):
                xcvr.disconnect()

            disconnect_mock.assert_called_once()
            medium.disconnect.assert_called_once_with(medium_conn)
            conn_client.send.assert_called_once_with(None)

            self.assertIsNone(xcvr._medium)

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
            conn_client_mock.send.assert_called_once_with(ManagementMessages.CLOSE)
        except AssertionError:
            raise

        terminate_mock.assert_called_once_with(xcvr)
        join_mock.assert_called_once()

        self.assertIsNone(Transceiver._instances.get((TestTransceiver, name)))

        # Calling stop() again should not raise an exception.
        xcvr.stop()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    @patch(f'{BASE_TARGET}.Process.terminate', autospec=True)
    @patch.object(MockTransceiver, 'join')
    def test_stop_broken_pipe(self, join_mock, terminate_mock, PipeMock, EventMock):
        PipeMock.side_effect = [
            (None, conn_client_mock := self.build_unique_mock('send')),
            (None, None),
        ]
        xcvr = self.build_xcvr(is_alive=True)
        name = xcvr.name
        conn_client_mock.send.side_effect = BrokenPipeError()

        xcvr.stop()

        terminate_mock.assert_called_once_with(xcvr)
        join_mock.assert_called_once()

        self.assertIsNone(Transceiver._instances.get((TestTransceiver, name)))

    def test_stop_in_practice(self):
        xcvr = self.build_xcvr(auto_start=True)

        self.assertFalse(xcvr._stop_event.is_set())

        xcvr.stop()
        self.assertIsNone(Transceiver._instances.get((TestTransceiver, xcvr.name)))
        self.assertTrue(xcvr._stop_event.is_set())

        # We can still call stop a second time without raising an exception.
        xcvr.stop()

    # endregion

    # region misc

    def test_init_shared_objects(self):
        test_val = 123

        class MyTransceiver(
            MockTransceiver, supported_media=[MockMedium], buffer_bytes=1
        ):
            def _init_shared_objects(self):
                self._test_val = Value('i', test_val)

        xcvr = MyTransceiver(name='foo', base_baud=100)
        self.assertEqual(xcvr._test_val.value, test_val)

    # endregion


class TestMisc(TestCase, ProcessBuilderMixin):
    def test_cleanup_processes(self):
        xcvrs = [Mock(), Mock()]
        medium = [Mock(), Mock()]

        # Make the first item in each list raise an AttributeError when its terminate() is
        # called.
        xcvrs[0].terminate.side_effect = AttributeError()
        medium[0].terminate.side_effect = AttributeError()

        # Add the mocks to their respective _instances class dictionaries.
        for i, x in enumerate(xcvrs):
            Transceiver._instances[i] = x
        for i, m in enumerate(medium):
            Medium._instances[i] = m

        # Call the cleanup function and observe that each of the mocks has had its
        # terminate() method called.
        cleanup_processes()

        for i, x in enumerate(xcvrs):
            x.terminate.assert_called_once()
            if i > 0:
                x.join.assert_called_once()
        for i, m in enumerate(medium):
            m.terminate.assert_called_once()
            if i > 0:
                m.join.assert_called_once()


# A couple of symbols that allow our rudimentary transceiver to know when it has
# reached the start and end of a transmission.
START_SYMBOL = 126
END_SYMBOL = 127

READ_MSG = 'READ!'


class IPCTransceiver(Transceiver, supported_media=[MockMedium], buffer_bytes=16):
    """A very rudinmentary transceiver that can be used for producing and consuming a
    single item from the tx and rx buffers.
    """

    def _init_shared_objects(self):
        self.rx_ongoing = Value('b', 0)
        self.tx_ongoing = Value('b', 0)

        self.rx_index = Value('b', 0)
        self.tx_index = Value('b', 0)

        # Normally there would be some sort of preamble to sync the receiver mechanism
        # with the transmission frequency, but we're not going to bother with that here.
        # Instead, we'll just store the last amplitude and use the fact that the value
        # will never be the same for two consecutive samples to determine boundaries.
        self.last_rx_value = Value('b', 0)

    def _process_rx_amplitude(self, amplitude, *args, **kwargs):
        # This is tricksy. We need to know when we've reached the end of one symbol and
        # the start of the next. The receiver starts off in a state where it's waiting for
        # the medium to be active before sampling, so we can use the first amplitude
        # received to indicate that the medium is active and we should start storing.
        if not self.rx_ongoing.value:
            if amplitude != START_SYMBOL:
                # Nothing to receive yet.
                return

            # Our buffer now contains data to receive.
            self.rx_ongoing.value = 1

        if amplitude == self.last_rx_value.value:
            # We're still looking at the same symbol as last time and it's not the
            # end symbol, so we can ignore this sample.
            return

        if self.last_rx_value.value == END_SYMBOL and amplitude == 0:
            # The end symbol has finished trasnmitting, so we can stop receiving.
            self.rx_ongoing.value = 0
            self.last_rx_value.value = 0

            # Alert the higher layer that it may now observe the buffer.
            self._osi_client.send(READ_MSG)

            # Tell the medium that we're waiting for the medium to stop being idle.
            return -1

        # Place the received amplitude in the buffer
        self._set_rx_buffer_element(int(amplitude), self.rx_index.value)

        self.rx_index.value += 1

        # Keep track of the last amplitude we received.
        self.last_rx_value.value = int(amplitude)

    def _next_tx_symbol(self, *args, **kwargs):
        symbol = self.get_tx_buffer_element(self.tx_index.value)

        if not self.tx_ongoing.value:
            if symbol != START_SYMBOL:
                # Nothing to transmit yet.
                return None

            # Our buffer now contains data to transmit.
            self.tx_ongoing.value = 1

        if symbol == END_SYMBOL:
            # We've reached the end of the transmission.
            self.tx_ongoing.value = 0

        self.tx_index.value += 1

        return symbol


class TestIPC(TestPHYBase):
    def test_send_data_between_transceivers(self):
        # Make sure a broadcast is received by all transceivers.
        names = {'alice', 'bob', 'carla', 'dave'}
        base_baud = 1e6
        xcvrs = {n: IPCTransceiver(name=n.title(), base_baud=base_baud) for n in names}

        tx_data_len = 10
        last_index = tx_data_len + 1

        medium = MockMedium(name='air')

        for xcvr in xcvrs.values():
            xcvr.connect(medium)

        # Every sender has their message heard by every receiver and does not hear their
        # own message.
        for i, sender_name in enumerate(names):
            receiver_names = names - {sender_name}

            sender = xcvrs[sender_name]

            # Place a sequence of values in the sender's transmit buffer, with the start
            # value being the index of the sender in the list of names.
            tx_data = list(range(i, i + tx_data_len))

            with sender._tx_buffer.get_lock():
                sender.set_tx_buffer_data([START_SYMBOL] + tx_data + [END_SYMBOL])

            with self.subTest(sender=sender_name, receivers=receiver_names):
                # Each of the receivers should have received the data.
                for name in receiver_names:
                    receiver = xcvrs[name]

                    if not receiver.osi_listener.poll(timeout=1):
                        self.fail(f'No data in {sender_name} to {name} test')

                    # Make sure that we got the read message.
                    self.assertEqual(receiver.osi_listener.recv(), READ_MSG)

                    # Observe and consume the data.
                    with receiver._rx_buffer.get_lock():
                        rx_data = receiver.get_rx_buffer_data(1, last_index)

                        self.assertEqual(rx_data, tx_data)

                        # Make sure the buffer is empty before moving on to the next test.
                        receiver.clear_rx_buffer()

                        receiver.rx_index.value = 0

                # The sender should not have received the data.
                self.assertFalse(sender.osi_listener.poll(timeout=0.1))

            # Clear the sender's transmit buffer.
            with sender._tx_buffer.get_lock():
                sender.clear_tx_buffer()

                sender.tx_index.value = 0

        # Make sure that the transceivers are stopped.
        for xcvr in xcvrs.values():
            xcvr.stop()


# endregion

if __name__ == '__main__':
    main()
