#!/usr/bin/env python
from __future__ import annotations

from math import sin
from multiprocessing import Value
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event, Lock
from unittest import TestCase, main
from unittest.mock import Mock, call, patch

from pynet.physical.base import (
    CommsManager,
    Medium,
    Transceiver,
    Transmission,
    cleanup_processes,
)
from pynet.physical.constants import (
    NANOSECONDS_PER_SECOND,
    SPEED_OF_LIGHT,
    TIME_DILATION_FACTOR,
    ManagementMessages,
    CommsType,
    Responses,
)
from pynet.physical.exceptions import (
    ConnectionError,
    ProcessNotRunningError,
    TransmissionComplete,
)
from pynet.testing import LogTestingMixin, ProcessBuilderMixin

BASE_TARGET = 'pynet.physical.base'

# region Helper Classes


class MockMedium(Medium, dimensionality=1, velocity_factor=0.77):
    """A helper class for testing the :class:`Medium` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _connect(self, *args, **kwargs) -> int:
        with self._comms_details_lock:
            return len(self._comms_details)


class MockTransceiver(Transceiver, supported_media=[MockMedium], buffer_bytes=1500):
    """A helper class for testing the :class:`Transceiver` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _process_rx_amplitude(self, amplitude, *args, **kwargs):
        return amplitude

    def _next_tx_symbol(self, *args, **kwargs):
        return 0


# endregion

# region Tests


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
        duration_ns = 200
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


class TestCommsManager(TestCase, LogTestingMixin):
    log_target: str = BASE_TARGET

    def setUp(self):
        super().setUp()
        self.cmgr = self._build_cmgr()

    def _build_cmgr(self, conn=None) -> CommsManager:
        return CommsManager(conn=conn or Mock())

    def test__init__(self):
        mock_conn = Mock()

        cmgr = CommsManager(conn=mock_conn)

        self.assertEqual(cmgr._conn, mock_conn)

    def test_send(self):
        cmgr = self.cmgr
        conn = cmgr._conn
        data = 1

        with self.subTest('BrokenPipe'):
            conn.send.side_effect = BrokenPipeError
            with self.assertRaises(BrokenPipeError) as ctx:
                cmgr._send(data)

            conn.send.assert_called_once_with(data)

        conn.reset_mock()

        with self.subTest('ConnectionError'):
            conn.send.side_effect = ConnectionError(err_str := 'test')
            with self.assertRaises(ConnectionError) as ctx:
                cmgr._send(data)

            self.assertEqual(
                str(ctx.exception),
                f'Could not send data to the Transceiver via {cmgr}: {err_str}',
            )
            conn.send.assert_called_once_with(data)

        conn.reset_mock()
        conn.send.side_effect = None

        with self.subTest('Success'):
            self.assertIsNone(cmgr._send(data))
            conn.send.assert_called_once_with(data)

    def test_recv(self):
        cmgr = self.cmgr
        conn = cmgr._conn

        with self.subTest('BrokenPipe'):
            conn.recv.side_effect = BrokenPipeError
            with self.assertRaises(BrokenPipeError) as ctx:
                cmgr._recv()

            conn.recv.assert_called_once_with()

        conn.reset_mock()

        with self.subTest('ConnectionError'):
            conn.recv.side_effect = ConnectionError(err_str := 'test')
            with self.assertRaises(ConnectionError) as ctx:
                cmgr._recv()

            self.assertEqual(
                str(ctx.exception),
                f'Could not receive data from the Transceiver via {cmgr}: {err_str}',
            )
            conn.recv.assert_called_once_with()

        conn.reset_mock()
        conn.recv.side_effect = None
        conn.recv.return_value = data = 1

        with self.subTest('Success'):
            self.assertEqual(cmgr._recv(), data)
            conn.recv.assert_called_once_with()

    def test_trigger_tx(self):
        cmgr = self.cmgr
        cmgr._conn.recv.return_value = (
            symbol_fn := Mock(),
            duration_ns := 5,
        )

        self.assertEqual(cmgr.trigger_tx(), (symbol_fn, duration_ns))
        cmgr._conn.send.assert_called_once_with((CommsType.TRANSMIT, None))
        cmgr._conn.recv.assert_called_once()

    @patch(f'{BASE_TARGET}.CommsManager._get_amplitude', autospec=True)
    def test_maybe_trigger_rx(self, amp_mock, *mocks):
        cmgr = self._build_cmgr()
        conn = cmgr._conn
        conn_send = conn.send
        conn_recv = conn.recv

        def reset_mocks():
            amp_mock.reset_mock()
            conn_send.reset_mock()
            conn_recv.reset_mock()

        with self.subTest('waiting for idle medium, medium is idle'):
            amp_mock.return_value = 0
            cmgr.next_rx_ns = -1
            current_time = 123

            cmgr.maybe_trigger_rx(current_time)

            amp_mock.assert_called_once_with(cmgr, current_time)
            conn_send.assert_not_called()
            self.assertEqual(cmgr.next_rx_ns, -1)

        reset_mocks()

        with self.subTest('waiting for idle medium, medium is not idle'):
            current_time = 100
            cmgr.next_rx_ns = -1
            amp_mock.return_value = amp = 1.2
            conn_recv.return_value = rx_delta_ns = 10

            cmgr.maybe_trigger_rx(current_time)

            amp_mock.assert_called_once_with(cmgr, current_time)
            conn_send.assert_called_once_with((CommsType.RECEIVE, amp))
            conn_recv.assert_called_once()
            self.assertEqual(
                cmgr.next_rx_ns, current_time + rx_delta_ns * TIME_DILATION_FACTOR
            )

        reset_mocks()

        with self.subTest(
            'Sampling, two ongoing transmissions plus one completed and one error'
        ):
            cmgr.next_rx_ns = orig_next_rx_ns = 100
            amp_mock.return_value = amp = 1.2
            conn_recv.return_value = rx_delta_ns = 10

            cmgr.maybe_trigger_rx(orig_next_rx_ns)

            amp_mock.assert_called_once_with(cmgr, orig_next_rx_ns)
            conn.send.assert_called_with((CommsType.RECEIVE, amp))
            conn_recv.assert_called_once()
            self.assertEqual(
                cmgr.next_rx_ns, orig_next_rx_ns + rx_delta_ns * TIME_DILATION_FACTOR
            )

        reset_mocks()

        with self.subTest('switching from sampling to idle'):
            cmgr.next_rx_ns = orig_next_rx_ns = 100
            amp_mock.return_value = amp = 0
            conn_recv.return_value = rx_delta_ns = -1

            cmgr.maybe_trigger_rx(orig_next_rx_ns)

            amp_mock.assert_called_once_with(cmgr, orig_next_rx_ns)
            conn.send.assert_called_with((CommsType.RECEIVE, amp))
            self.assertEqual(cmgr.next_rx_ns, -1)

    def test_get_amplitude(self):
        cmgr = self.cmgr

        tx1 = Mock()
        tx2 = Mock()
        tx3 = Mock()
        tx4 = Mock()

        tx3.get_amplitude.side_effect = TransmissionComplete
        tx4.get_amplitude.side_effect = ValueError(err_str := 'test')

        cmgr._transmissions = {tx1, tx2, tx3, tx4}

        with self.assertInTargetLogs(
            'ERROR',
            f'{cmgr}: an exception occurred while processing a transmission: {err_str}',
        ):
            cmgr._get_amplitude(123)

        self.assertEqual(cmgr._transmissions, {tx1, tx2, tx4})

    @patch(f'{BASE_TARGET}.Transmission', autospec=True)
    def test_add_transmission(self, TransmissionMock):
        cmgr = self.cmgr

        symbol_fn = sin
        start_ns = 100
        duration_ns = 200
        attenuation = 0.5

        cmgr.add_transmission(symbol_fn, start_ns, duration_ns, attenuation)

        TransmissionMock.assert_called_once_with(
            symbol_fn, start_ns, duration_ns, attenuation
        )
        self.assertEqual(cmgr._transmissions, {TransmissionMock.return_value})


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

    @patch.object(MockMedium, 'start')
    def test__init__(self, mock_start):
        for auto_start in (True, False):
            with self.subTest(auto_start=auto_start):
                medium = self.build_medium(auto_start=auto_start)

                self.assertEqual(medium._dimensionality, MockMedium._dimensionality)
                self.assertEqual(medium._comms_details, {})
                self.assertEqual(medium._transmission_completion_times, {})

                self.assertIsInstance(medium._comms_details_lock, Lock)
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

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Thread', autospec=True)
    @patch.object(MockMedium, '_process_medium')
    def test_run(self, process_mock, ThreadMock, *mocks):
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
        )

        process_mock.assert_called_once()
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once()

    # region _monitor_connections-related

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_remove_connection')
    @patch.object(MockMedium, '_add_connection')
    def test_monitor_connections(self, add_mock, remove_mock, *mocks):
        medium = self.build_medium()

        conn_queue_get = medium._connection_queue.get
        stop_is_set = medium._stop_event.is_set

        def reset_mocks():
            conn_queue_get.reset_mock()
            conn_queue_get.side_effect = None
            stop_is_set.reset_mock()
            add_mock.reset_mock()
            remove_mock.reset_mock()

        with self.subTest('close message'):
            stop_is_set.side_effect = [False]
            conn_queue_get.return_value = ManagementMessages.CLOSE

            medium._monitor_connections()

            conn_queue_get.assert_called_once()
            stop_is_set.assert_called_once()
            add_mock.assert_not_called()
            remove_mock.assert_not_called()

        reset_mocks()

        with self.subTest('bad connection request'):
            stop_is_set.side_effect = [False, True]
            conn_queue_get.return_value = 'bad request'

            with self.assertInTargetLogs(
                'ERROR', f'{medium}: unexpected connection details type: str'
            ):
                medium._monitor_connections()

            conn_queue_get.assert_called_once()
            self.assertEqual(stop_is_set.call_count, 2)
            add_mock.assert_not_called()
            remove_mock.assert_not_called()

        reset_mocks()

        with self.subTest('add connection'):
            stop_is_set.side_effect = [False, True]
            conn_queue_get.return_value = creq = Mock(create=True)

            medium._monitor_connections()

            conn_queue_get.assert_called_once()
            self.assertEqual(stop_is_set.call_count, 2)
            add_mock.assert_called_once_with(creq)
            remove_mock.assert_not_called()

        reset_mocks()

        with self.subTest('remove connection'):
            stop_is_set.side_effect = [False, True]
            conn_queue_get.return_value = creq = Mock(create=False)

            medium._monitor_connections()

            conn_queue_get.assert_called_once()
            self.assertEqual(stop_is_set.call_count, 2)
            add_mock.assert_not_called()
            remove_mock.assert_called_once_with(creq)

    @patch(f'{BASE_TARGET}.CommsManager', autospec=True)
    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    @patch.object(MockMedium, '_connect')
    def test_add_connection(self, connect_mock, mono_mock, CommsManagerMock):
        medium = self.build_medium()
        mock_conn = Mock()
        conn_req = Mock(conn=mock_conn)
        conn_send = mock_conn.send
        location = 123

        def reset_mocks():
            connect_mock.reset_mock()
            mono_mock.reset_mock()
            CommsManagerMock.reset_mock()
            mock_conn.reset_mock()
            conn_send.reset_mock()

            connect_mock.side_effect = None
            conn_send.side_effect = None

        with self.subTest('connect exception'):
            connect_mock.side_effect = err = Exception(err_str := 'connect exception')

            with self.assertInTargetLogs(
                'ERROR',
                f'{medium}: connection failed: {err_str}',
            ):
                medium._add_connection(conn_req)

            connect_mock.assert_called_once_with(conn_req)
            conn_send.assert_called_once_with((Responses.ERROR, err))
            CommsManagerMock.assert_not_called()

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
            CommsManagerMock.assert_called_once_with(mock_conn)
            self.assertNotIn(location, medium._comms_details)

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
            CommsManagerMock.assert_called_once_with(mock_conn)
            self.assertEqual(
                medium._comms_details[location], CommsManagerMock.return_value
            )

    def test_remove_connection(self, *mocks):
        medium = self.build_medium()

        location = 123
        conn = Mock()
        creq = Mock(location=location)
        cmgr = Mock(_conn=conn)

        def reset_mocks():
            disconnect_mock.reset_mock()
            disconnect_mock.side_effect = None
            conn.reset_mock()

        with self.subTest('disconnect error'):
            with patch.object(medium, '_disconnect') as disconnect_mock:
                medium._comms_details = {location: cmgr}
                disconnect_mock.side_effect = Exception(err_str := 'disconnect exception')

                with self.assertInTargetLogs(
                    'ERROR',
                    f'{medium}: subclass disconnection failed: {err_str}. Continuing with removal.',
                ):
                    medium._remove_connection(creq)

                disconnect_mock.assert_called_once_with(creq)
                conn.close.assert_called_once()
                self.assertNotIn(location, medium._comms_details)

        reset_mocks()

        with self.subTest('success'):
            with patch.object(medium, '_disconnect') as disconnect_mock:
                medium._comms_details = {location: cmgr}

                with self.assertInTargetLogs(
                    'INFO',
                    f'{medium}: closing connection at {location=}',
                ):
                    medium._remove_connection(creq)

                disconnect_mock.assert_called_once_with(creq)
                conn.close.assert_called_once()
                self.assertNotIn(location, medium._comms_details)

        reset_mocks()

        with self.subTest('missing disconnect is okay'):
            # Here, we use the original _disconnect, which is a no-op.
            medium._comms_details = {location: cmgr}

            with self.assertInTargetLogs(
                'INFO',
                f'{medium}: closing connection at {location=}',
            ):
                medium._remove_connection(creq)

            conn.close.assert_called_once()
            self.assertNotIn(location, medium._comms_details)

    # endregion

    # region _process_medium-related

    @patch.object(MockMedium, '_process_rx_events')
    @patch.object(MockMedium, '_process_tx_events')
    @patch(f'{BASE_TARGET}.monotonic_ns')
    def test_process_medium(self, monotonic_ns_mock, process_tx_mock, process_rx_mock):
        medium = self.build_medium()

        loc1 = 123
        loc2 = 456

        cmgr1 = Mock()
        cmgr2 = Mock()

        medium._comms_details = {loc1: cmgr1, loc2: cmgr2}

        # Run _process_medium() confirming that we've collected the current time, that we
        # have triggered a potential advancing of buffers and that each connection has
        # been processed.
        medium._process_medium()

        current_ns = monotonic_ns_mock.return_value

        monotonic_ns_mock.assert_called_once()
        for process_mock in (process_tx_mock, process_rx_mock):
            process_mock.assert_has_calls(
                [call(loc1, cmgr1, current_ns), call(loc2, cmgr2, current_ns)]
            )

    @patch.object(MockMedium, '_calculate_travel_time_ns')
    def test_process_tx_events(self, calc_travel_time_mock):
        medium = self.build_medium()

        symbol_fn = Mock()
        current_ns = 500
        duration_ns = 100

        src_loc = 123
        src_cmgr = Mock(next_rx_ns=current_ns - 1)

        dest_loc = 456
        dest_cmgr = Mock()

        src_trigger_tx = src_cmgr.trigger_tx
        src_add_tranmsission = src_cmgr.add_transmission
        dest_add_tranmsission = dest_cmgr.add_transmission

        def reset_mocks():
            src_cmgr.reset_mock()
            src_trigger_tx.reset_mock()
            calc_travel_time_mock.reset_mock()
            src_add_tranmsission.reset_mock()
            dest_add_tranmsission.reset_mock()

            src_trigger_tx.side_effect = None

        src_trigger_tx.return_value = None

        with self.subTest('tx triggered with no symbol'):
            medium._process_tx_events(src_loc, src_cmgr, current_ns)

            src_trigger_tx.assert_called_once()

            src_add_tranmsission.assert_not_called()
            dest_add_tranmsission.assert_not_called()

        reset_mocks()

        medium._comms_details = {dest_loc: dest_cmgr, src_loc: src_cmgr}

        with self.subTest('tx triggered for first symbol'):
            # No ongoing tranmissions
            medium._transmission_completion_times = {}

            src_trigger_tx.return_value = symbol_fn, duration_ns
            calc_travel_time_mock.return_value = prop_delay_ns = 123

            with self.assertInTargetLogs(
                'DEBUG',
                f'{medium}: transceiver at location={src_loc} beginning transmission of new symbol',
            ):
                medium._process_tx_events(src_loc, src_cmgr, current_ns)

            src_trigger_tx.assert_called_once()

            self.assertEqual(
                medium._transmission_completion_times,
                {src_cmgr: current_ns + duration_ns * TIME_DILATION_FACTOR},
            )

            calc_travel_time_mock.assert_called_once_with(src_loc, dest_loc)

            dest_add_tranmsission.assert_called_once_with(
                symbol_fn, current_ns + prop_delay_ns, duration_ns, 1
            )
            src_add_tranmsission.assert_not_called()

        reset_mocks()

        with self.subTest('tx triggered for next symbol'):
            # A transmission that recently completed.
            stop_ns = current_ns - 30
            medium._transmission_completion_times = {src_cmgr: stop_ns}

            with self.assertInTargetLogs(
                'DEBUG',
                f'{medium}: transceiver at location={src_loc} beginning transmission of new symbol',
            ):
                medium._process_tx_events(src_loc, src_cmgr, current_ns)

            src_trigger_tx.assert_called_once()

            self.assertEqual(
                medium._transmission_completion_times,
                {src_cmgr: stop_ns + duration_ns * TIME_DILATION_FACTOR},
            )

            calc_travel_time_mock.assert_called_once_with(src_loc, dest_loc)

            # Also make sure that the destination connection manager has received the
            # symbol (and that the source has not).
            dest_add_tranmsission.assert_called_once_with(
                symbol_fn, stop_ns + prop_delay_ns, duration_ns, 1
            )
            src_add_tranmsission.assert_not_called()

        reset_mocks()

        with self.subTest('tx ongoing'):
            current_ns = stop_ns - 1

            medium._process_tx_events(src_loc, src_cmgr, current_ns)

            src_trigger_tx.assert_not_called()
            calc_travel_time_mock.assert_not_called()
            dest_add_tranmsission.assert_not_called()
            src_add_tranmsission.assert_not_called()

        reset_mocks()

        with self.subTest('tx triggered with broken pipe'):
            medium._transmission_completion_times = {}
            current_ns = stop_ns + duration_ns + 1

            src_trigger_tx.side_effect = BrokenPipeError(err_str := 'test')

            with self.assertInTargetLogs(
                'DEBUG',
                f'{medium}: connection to transceiver at {src_loc} closed: {err_str}',
            ):
                medium._process_tx_events(src_loc, src_cmgr, current_ns)

            src_trigger_tx.assert_called_once()

        reset_mocks()

        with self.subTest('tx triggered with connection error'):
            medium._transmission_completion_times = {}
            current_ns = stop_ns + duration_ns + 1

            src_trigger_tx.side_effect = ConnectionError(err_str := 'test')

            with self.assertInTargetLogs(
                'ERROR', f'{medium}: error while triggering tx: {err_str}'
            ):
                medium._process_tx_events(src_loc, src_cmgr, current_ns)

            src_trigger_tx.assert_called_once()

    def test_process_rx_events(self):
        medium = self.build_medium()
        current_ns = 500

        loc = 123
        cmgr = Mock()
        maybe_trigger_rx = cmgr.maybe_trigger_rx

        def reset_mocks():
            cmgr.reset_mock()
            maybe_trigger_rx.reset_mock()
            maybe_trigger_rx.side_effect = None

        with self.subTest('rx not triggered'):
            # No ongoing tranmissions
            cmgr.next_rx_ns = current_ns + 1

            with self.assertNotInTargetLogs(
                'DEBUG',
                f'{medium}: maybe triggering rx at location={loc}',
            ):
                medium._process_rx_events(loc, cmgr, current_ns)

            maybe_trigger_rx.assert_not_called()

        reset_mocks()

        with self.subTest('rx triggered, broken pipe'):
            cmgr.next_rx_ns = current_ns - 1
            maybe_trigger_rx.side_effect = BrokenPipeError(err_str := 'test')

            with self.assertInTargetLogs(
                'DEBUG',
                (
                    f'{medium}: maybe triggering rx at location={loc}',
                    f'{medium}: connection to transceiver at {loc} closed: {err_str}',
                ),
            ):
                medium._process_rx_events(loc, cmgr, current_ns)

            maybe_trigger_rx.assert_called_once_with(current_ns)

        reset_mocks()

        with self.subTest('rx triggered, connection error'):
            maybe_trigger_rx.side_effect = ConnectionError(err_str := 'test')

            with self.assertInTargetLogs(
                'ERROR',
                f'{medium}: error while triggering rx: {err_str}',
            ):
                medium._process_rx_events(loc, cmgr, current_ns)

            maybe_trigger_rx.assert_called_once_with(current_ns)

    def test_calculate_travel_time_ns(self):
        medium = self.build_medium()
        meters = medium._medium_velocity_ns

        # Test for 1D and 3D, with each having been set up to have a distance equal to
        # the number of meters that a signal travels per nanosecond in the medium.
        locs = (((0, 0, 0), (0, 0, meters)), (0, meters))

        for loc1, loc2 in locs:
            with self.subTest(loc1=loc1, loc2=loc2):
                self.assertEqual(medium._calculate_travel_time_ns(loc1, loc2), 1)

    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    def test_transmission_and_reception(self, monotonic_mock, *mocks):
        """
        Simulate a transmission triggered in a source received by a destination at the
        right time.
        """
        medium = self.build_medium()
        current_time = 0
        symbol_fn = sin
        duration_ns = 100

        # The details about the transceivers.
        src_loc = 100
        dest_loc = 105
        src_conn = Mock()
        src_send = src_conn.send
        src_recv = src_conn.recv
        dest_conn = Mock()
        dest_send = dest_conn.send
        dest_recv = dest_conn.recv

        travel_time_ns = medium._calculate_travel_time_ns(src_loc, dest_loc)
        start_ns = current_time + travel_time_ns
        stop_ns = start_ns + duration_ns * TIME_DILATION_FACTOR

        # Configure the connection managers such that all rx events are upcoming.
        src_cmgr = CommsManager(src_conn)
        dest_cmgr = CommsManager(dest_conn)
        medium._comms_details = {src_loc: src_cmgr, dest_loc: dest_cmgr}

        def reset_mocks():
            src_conn.reset_mock()
            dest_conn.reset_mock()
            src_send.reset_mock()
            src_recv.reset_mock()
            dest_send.reset_mock()
            dest_recv.reset_mock()

        with self.subTest('Transmission picked up from source to destination'):
            monotonic_mock.return_value = current_time

            # Prepare the transmission for the source and configure the destination to not
            # have any ongoing transmissions.
            src_recv.return_value = (symbol_fn, duration_ns)
            dest_recv.return_value = None

            medium._process_medium()

            src_send.assert_called_once_with((CommsType.TRANSMIT, None))
            dest_send.assert_called_once_with((CommsType.TRANSMIT, None))
            src_recv.assert_called_once()
            dest_recv.assert_called_once()

            # The confirm that the set of transmissions at the source is still empty.
            self.assertEqual(src_cmgr._transmissions, set())

            # Meanwhile, the transmissions for the destination have been updated with a
            # new transmission that has the expected details.
            self.assertEqual(len(dest_cmgr._transmissions), 1)

            dest_transmission = list(dest_cmgr._transmissions)[0]
            self.assertEqual(dest_transmission._symbol_fn, symbol_fn)
            self.assertEqual(dest_transmission._start_ns, start_ns)
            self.assertEqual(dest_transmission._stop_ns, stop_ns)
            self.assertEqual(dest_transmission._attenuation, 1)

        reset_mocks()

        with self.subTest('Transmission still incoming'):
            # We now advance time a bit, but not enough for the symbol to start being
            # received by the destination.
            monotonic_mock.return_value = current_time = start_ns - 1

            # Configure the destination to be ready to sample the medium.
            dest_cmgr.next_rx_ns = current_time - 1

            # Ignore reception for the source.
            src_cmgr.next_rx_ns = current_time + 1

            dest_next_rx_delta_ns = 100

            # Configure the source and destination such that neither has any new
            # transmissions.
            src_recv.return_value = None

            # The destination will have a reception triggered and a check for a
            # transmission, so both send() and recv() will be called twice. The first
            # recv() should return None for no transmission, and the second should return
            # the delta for the next reception time.
            dest_recv.side_effect = [None, dest_next_rx_delta_ns]

            medium._process_medium()

            # No signal has reached the destination yet.
            dest_send.assert_has_calls(
                [call((CommsType.TRANSMIT, None)), call((CommsType.RECEIVE, 0.0))]
            )
            self.assertEqual(dest_recv.call_count, 2)

        reset_mocks()

        with self.subTest('Transmission starts being received'):
            # We now advance time such that the symbol should be received by the
            # destination.
            monotonic_mock.return_value = current_time = (
                current_time + dest_next_rx_delta_ns + 1
            )

            # Configure the destination to be ready to sample the medium.
            dest_cmgr.next_rx_ns = orig_dest_rx = current_time - 1

            # Ignore reception for the source.
            src_cmgr.next_rx_ns = current_time + 1

            # Configure the source and destination such that neither has any new
            # transmissions.
            src_recv.return_value = None

            # The destination will have a reception triggered and a check for a
            # transmission, so both send() and recv() will be called twice. The first
            # recv() should return None for no transmission, and the second should return
            # the delta for the next reception time.
            dest_recv.side_effect = [None, dest_next_rx_delta_ns]

            medium._process_medium()

            # The destination should have begun receiving the symbol.
            amplitude = symbol_fn((orig_dest_rx - start_ns) / TIME_DILATION_FACTOR)
            dest_send.assert_has_calls(
                [call((CommsType.TRANSMIT, None)), call((CommsType.RECEIVE, amplitude))]
            )
            self.assertEqual(dest_recv.call_count, 2)

        reset_mocks()

        with self.subTest('Transmission complete'):
            # We now advance time such that the symbol is finished from the perspective of
            # the destination.
            monotonic_mock.return_value = current_time = stop_ns + 1

            # Configure the destination to be ready to sample the medium.
            dest_cmgr.next_rx_ns = orig_dest_rx = current_time - 1

            # Ignore reception for the source.
            src_cmgr.next_rx_ns = current_time + 1

            # Configure the source and destination such that neither has any new
            # transmissions.
            src_recv.return_value = None

            # The destination will have a reception triggered and a check for a
            # transmission, so both send() and recv() will be called twice. The first
            # recv() should return None for no transmission, and the second should return
            # the delta for the next reception time.
            dest_recv.side_effect = [None, dest_next_rx_delta_ns]

            medium._process_medium()

            dest_send.assert_has_calls(
                [call((CommsType.TRANSMIT, None)), call((CommsType.RECEIVE, 0.0))]
            )
            self.assertEqual(dest_recv.call_count, 2)

            # There are no transmissions in sight anywhere.
            self.assertEqual(src_cmgr._transmissions, set())
            self.assertEqual(dest_cmgr._transmissions, set())
            self.assertEqual(medium._transmission_completion_times, {})

    # endregion

    # region Public Methods

    @patch(f'{BASE_TARGET}.Process.terminate', autospec=True)
    @patch.object(MockMedium, 'join')
    def test_stop_in_theory(self, join_mock, terminate_mock):
        medium = self.build_medium()
        medium._stop_event = stop_event = Mock()
        medium._connection_queue = conn_queue = Mock()
        medium.is_alive = Mock(return_value=True)

        with self.subTest('Queue is closed'):
            conn_queue.put.side_effect = ValueError
            medium.stop()

            stop_event.set.assert_called_once()
            conn_queue.put.assert_called_once_with(ManagementMessages.CLOSE)

            join_mock.assert_called_once()
            terminate_mock.assert_called_once_with(medium)

            self.assertIsNone(Medium._instances.get((TestMedium, medium.name)))

            medium.stop()

        # Build another medium for the second subtest
        medium = self.build_medium()
        medium._stop_event = stop_event = Mock()
        medium._connection_queue = conn_queue = Mock()
        medium.is_alive = Mock(return_value=True)
        join_mock.reset_mock()
        terminate_mock.reset_mock()

        with self.subTest('Queue is still open'):
            medium.stop()

            stop_event.set.assert_called_once()
            conn_queue.put.assert_called_once_with(ManagementMessages.CLOSE)

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


class TestTransceiver(TestPHYBase):
    def setUp(self):
        super().setUp()

        self.medium_mock = self.build_medium(mocked=True)

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
    def test__init__(self, PipeMock, mock_start):
        mod_listener = Mock()
        mod_client = Mock()
        osi_listener = Mock()
        osi_client = Mock()

        base_baud = 1e6

        for auto_start in (True, False):
            PipeMock.side_effect = [
                (mod_listener, mod_client),
                (osi_listener, osi_client),
            ]
            with self.subTest(auto_start=auto_start):

                xcvr = self.build_xcvr(auto_start=auto_start, base_baud=base_baud)

                PipeMock.assert_has_calls(
                    [
                        call(duplex=False),
                        call(duplex=False),
                    ]
                )

                self.assertIsNone(xcvr._medium)
                self.assertEqual(xcvr._base_delta_ns, NANOSECONDS_PER_SECOND // base_baud)
                self.assertEqual(xcvr._connections_listener, mod_listener)
                self.assertEqual(xcvr._connections_client, mod_client)
                self.assertEqual(xcvr.osi_listener, osi_listener)
                self.assertEqual(xcvr._osi_client, osi_client)

                self.assertEqual(list(xcvr._tx_buffer), [0] * xcvr._buffer_bytes)
                self.assertEqual(list(xcvr._rx_buffer), [0] * xcvr._buffer_bytes)

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

                def _next_tx_symbol(self):
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

                def _next_tx_symbol(self):
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

                def _next_tx_symbol(self):
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

                def _next_tx_symbol(self):
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
        medium.connect.side_effect = OSError(err_str)

        with patch(f'{BASE_TARGET}.Pipe', autospec=True) as PipeMock:
            PipeMock.return_value = (None, medium_conn := Mock())

            with self.assertRaisesRegex(
                ConnectionError,
                f'Error sending connection request to {medium!r}: {err_str}',
            ):
                xcvr.connect(medium)

        medium.connect.assert_called_once_with(medium_conn)
        PipeMock.assert_called_once_with(duplex=True)

    def test_connect_bad_recv(self):
        xcvr = self.build_xcvr(is_alive=True)
        medium = self.build_medium(mocked=True, is_alive=True)

        err_str = 'bar'

        with patch(f'{BASE_TARGET}.Pipe', autospec=True) as PipeMock:
            PipeMock.return_value = (xvcr_conn := Mock(), None)

            xvcr_conn.recv.side_effect = OSError(err_str)

            with self.assertRaisesRegex(
                ConnectionError,
                f'Error receiving connection response from {medium!r}: {err_str}',
            ):
                xcvr.connect(medium)

            xvcr_conn.recv.assert_called_once()

    def test_connect_bad_response_contents(self):
        xcvr = self.build_xcvr(is_alive=True)
        medium = self.build_medium(mocked=True, is_alive=True)

        with patch(f'{BASE_TARGET}.Pipe', autospec=True) as PipeMock:
            PipeMock.return_value = (xvcr_conn := Mock(), None)

            xvcr_conn.recv.return_value = response = 'foo'

            with self.assertRaisesRegex(
                ConnectionError,
                f'Unexpected response contents from {medium!r}: {response!r}',
            ):
                xcvr.connect(medium)

    def test_connect_rejected(self):
        xcvr = self.build_xcvr(is_alive=True)
        medium = self.build_medium(mocked=True, is_alive=True)

        with patch(f'{BASE_TARGET}.Pipe', autospec=True) as PipeMock:
            PipeMock.return_value = (xvcr_conn := Mock(), None)

            xvcr_conn.recv.return_value = (
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

        self.medium_mock.is_alive.return_value = True
        xcvr = self.build_xcvr()

        PipeMock.assert_has_calls([call(duplex=False), call(duplex=False)])
        PipeMock.reset_mock()

        # ...and one more in the connect method.
        PipeMock.side_effect = [(xcvr_conn := Mock(), medium_conn := Mock())]

        xcvr_conn.recv.return_value = (Responses.OK, (location := 123))

        with patch.object(xcvr, 'is_alive', return_value=True):
            with self.assertInTargetLogs(
                'DEBUG', f'Connecting {xcvr!r} to {self.medium_mock!r}'
            ):
                xcvr.connect(self.medium_mock, **kwargs)

        connect_mock.assert_called_once_with(self.medium_mock, **kwargs)
        PipeMock.assert_called_once_with(duplex=True)
        self.medium_mock.connect.assert_called_once_with(medium_conn, **kwargs)
        conn_client_mock.send.assert_called_once_with(xcvr_conn)
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

        # Build a pre-connected Transceiver.
        xcvr = self.build_xcvr()
        xcvr.location = location = 123
        xcvr._medium = self.medium_mock

        with self.assertInTargetLogs(
            'DEBUG', f'Disconnecting {xcvr!r} from {self.medium_mock!r}'
        ):
            xcvr.disconnect()

        disconnect_mock.assert_called_once()
        self.medium_mock.disconnect.assert_called_once_with(location)
        conn_client_mock.send.assert_called_once_with(None)

        self.assertIsNone(xcvr._medium)

    # endregion

    # region run()-related

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Thread', autospec=True)
    @patch(f'{BASE_TARGET}.Pipe', autospec=True)
    def test_run(self, PipeMock, ThreadMock, EventMock):
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
            ThreadMock.assert_called_once_with(
                target=xcvr._monitor_medium,
                args=(conn,),
            )
            mock_thread.start.assert_called_once()

            # Disconnect
            conn.send.assert_called_once_with(ManagementMessages.CLOSE)
            mock_thread.join.assert_called_once()

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
            ThreadMock.assert_called_once_with(
                target=xcvr._monitor_medium,
                args=(conn,),
            )
            mock_thread.start.assert_called_once()

            # Close
            conn.send.assert_not_called()
            mock_thread.join.assert_called_once()

    @patch.object(MockTransceiver, '_process_rx_amplitude')
    def test_process_reception_event(self, process_mock):
        xcvr = self.build_xcvr()
        conn = self.build_unique_mock('send')
        amplitude = 1.3

        def reset_mocks():
            process_mock.reset_mock()
            conn.send.reset_mock()
            process_mock.side_effect = None

        with self.subTest('Error raised'):
            process_mock.side_effect = ValueError(err_str := 'ruh roh')

            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error processing {amplitude=}: {err_str}',
            ):
                xcvr._process_reception_event(conn, amplitude)

            conn.send.assert_called_once_with(xcvr._base_delta_ns)

            process_mock.assert_called_once_with(amplitude)

        reset_mocks()

        with self.subTest('Sampling rate provided'):
            process_mock.return_value = delta = 123

            xcvr._process_reception_event(conn, amplitude)

            conn.send.assert_called_with(delta)

            process_mock.assert_called_once_with(amplitude)

        reset_mocks()

        with self.subTest('No sampling rate provided'):
            process_mock.return_value = None

            xcvr._process_reception_event(conn, amplitude)

            conn.send.assert_called_with(xcvr._base_delta_ns / 2)

            process_mock.assert_called_once_with(amplitude)

    @patch.object(MockTransceiver, '_next_tx_symbol')
    def test_process_transmission_event(self, next_mock):
        xcvr = self.build_xcvr()
        conn = self.build_unique_mock('send')
        send = conn.send
        base_delta_ns = xcvr._base_delta_ns

        def reset_mocks():
            next_mock.reset_mock()
            send.reset_mock()
            next_mock.side_effect = None
            conn.send.side_effect = None

        next_mock.side_effect = ValueError(err_str := 'bad next')

        with self.subTest('Failed to get next symbol'):
            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error getting next symbol to transmit: {err_str}',
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once_with(None)

        reset_mocks()

        with self.subTest('Bad symbol function'):
            next_mock.return_value = symbol_fn = Mock(
                side_effect=ValueError(err_str := 'bad symbol')
            )

            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error getting amplitude from symbol function: {err_str}',
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            symbol_fn.assert_called_once_with(0)
            send.assert_called_once_with(None)

        reset_mocks()

        with self.subTest('Good symbol function, bad return type'):
            amplitude = 'hi'
            next_mock.return_value = symbol_fn = Mock(return_value=amplitude)

            with self.assertInTargetLogs(
                'ERROR',
                (
                    f'{xcvr}: Symbol function returned an amplitude of type '
                    f'str ({amplitude!r}); expected int or float'
                ),
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            symbol_fn.assert_called_once_with(0)
            send.assert_called_once_with(None)

        reset_mocks()

        with self.subTest('Good symbol function, good return type'):
            value = 1.3
            next_mock.return_value = symbol_fn = Mock(return_value=value)

            xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            symbol_fn.assert_called_once_with(0)
            send.assert_called_once_with((symbol_fn, base_delta_ns))

        reset_mocks()

        with self.subTest('Bad non-function'):
            next_mock.return_value = symbol = 'hi'

            with self.assertInTargetLogs(
                'ERROR',
                (
                    f'{xcvr}: Symbol is of type str ({symbol!r}); expected int, float or '
                    'callable'
                ),
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once_with(None)

        reset_mocks()

        with self.subTest('Good non-function'):
            next_mock.return_value = symbol = 1.3

            xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once()

            send.assert_called_once_with((symbol, base_delta_ns))

        reset_mocks()

        with self.subTest('Error raised during send'):
            next_mock.return_value = symbol_fn = Mock(return_value=1)
            conn.send.side_effect = ValueError(err_str := 'bad send')

            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error sending symbol function to medium: {err_str}',
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once_with((symbol_fn, base_delta_ns))

        reset_mocks()

        with self.subTest('no symbol to send'):
            next_mock.return_value = None

            xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once_with(None)

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch.object(MockTransceiver, '_process_reception_event')
    @patch.object(MockTransceiver, '_process_transmission_event')
    def test_monitor_medium(self, tx_mock, rx_mock, EventMock):
        stop_event_mock = EventMock.return_value
        xcvr = self.build_xcvr()

        conn = self.build_unique_mock('recv')

        def reset_mocks():
            conn.recv.reset_mock()
            conn.recv.side_effect = None
            conn.close.reset_mock()
            stop_event_mock.is_set.reset_mock()
            rx_mock.reset_mock()
            tx_mock.reset_mock()

        with self.subTest('Error raised during initial receive'):
            stop_event_mock.is_set.side_effect = [False, True]
            conn.recv.side_effect = ValueError(err_str := 'bad recv')

            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error receiving data from medium: {err_str}',
            ):
                xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

        reset_mocks()

        with self.subTest('Close message received'):
            # A second call will throw an exception.
            stop_event_mock.is_set.side_effect = [False]
            conn.recv.return_value = ManagementMessages.CLOSE

            xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 1)

        reset_mocks()

        with self.subTest('Bad data sent from medium'):
            conn.recv.return_value = data = 123
            stop_event_mock.is_set.side_effect = [False, True]

            with self.assertInTargetLogs(
                'ERROR',
                (
                    f'{xcvr}: Received malformed communications event data from medium: '
                    f'{data=}'
                ),
            ):
                xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

        reset_mocks()

        with self.subTest('Receive event'):
            conn.recv.return_value = CommsType.RECEIVE, data
            stop_event_mock.is_set.side_effect = [False, True]

            xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

            rx_mock.assert_called_once_with(conn, data)
            tx_mock.assert_not_called()

        reset_mocks()

        with self.subTest('Transmit event'):
            conn.recv.return_value = CommsType.TRANSMIT, data
            stop_event_mock.is_set.side_effect = [False, True]

            xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

            tx_mock.assert_called_once_with(conn)
            rx_mock.assert_not_called()

        reset_mocks()

        with self.subTest('Bad comms type'):
            comms_type = 123
            conn.recv.return_value = comms_type, data
            stop_event_mock.is_set.side_effect = [False, True]

            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Received unknown communications event type from medium: '
                f'{comms_type=}',
            ):
                xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

            tx_mock.assert_not_called()
            rx_mock.assert_not_called()

    # endregion

    # region other public methods

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

    def __init__(self, *args, **kwargs):
        self.rx_ongoing = Value('b', 0)
        self.tx_ongoing = Value('b', 0)

        self.rx_index = Value('b', 0)
        self.tx_index = Value('b', 0)

        # Normally there would be some sort of preamble to sync the receiver mechanism
        # with the transmission frequency, but we're not going to bother with that here.
        # Instead, we'll just store the last amplitude and use the fact that the value
        # will never be the same for two consecutive samples to determine boundaries.
        self.last_rx_value = Value('b', 0)

        super().__init__(*args, **kwargs)

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
        with self._rx_buffer.get_lock():
            self._rx_buffer[self.rx_index.value] = int(amplitude)

        self.rx_index.value += 1

        # Keep track of the last amplitude we received.
        self.last_rx_value.value = int(amplitude)

    def _next_tx_symbol(self, *args, **kwargs):
        with self._tx_buffer.get_lock():
            symbol = self._tx_buffer[self.tx_index.value]

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
        xvcrs = {n: IPCTransceiver(name=n.title(), base_baud=base_baud) for n in names}

        tx_data_len = 10
        last_index = tx_data_len + 1

        medium = MockMedium(name='air')

        for xcvr in xvcrs.values():
            xcvr.connect(medium)

        # Every sender has their message heard by every receiver and does not hear their
        # own message.
        for i, sender_name in enumerate(names):
            receiver_names = names - {sender_name}

            sender = xvcrs[sender_name]

            # Place a sequence of values in the sender's transmit buffer, with the start
            # value being the index of the sender in the list of names.
            tx_data = list(range(i, i + tx_data_len))

            with sender._tx_buffer.get_lock():
                sender._tx_buffer[0] = START_SYMBOL
                for j in range(tx_data_len):
                    sender._tx_buffer[j + 1] = tx_data[j]
                sender._tx_buffer[j + 2] = END_SYMBOL

            with self.subTest(sender=sender_name, receivers=receiver_names):
                # Each of the receivers should have received the data.
                for name in receiver_names:
                    receiver = xvcrs[name]

                    if not receiver.osi_listener.poll(timeout=1):
                        self.fail(f'No data in {sender_name} to {name} test')

                    # Make sure that we got the read message.
                    self.assertEqual(receiver.osi_listener.recv(), READ_MSG)

                    # Observe and consume the data.
                    with receiver._rx_buffer.get_lock():
                        rx_data = receiver._rx_buffer[1:last_index]

                        self.assertEqual(rx_data, tx_data)

                        # Make sure the buffer is empty before moving on to the next test.
                        for j in range(tx_data_len + 2):
                            receiver._rx_buffer[j] = 0

                        receiver.rx_index.value = 0

                # The sender should not have received the data.
                self.assertFalse(sender.osi_listener.poll(timeout=0.1))

            # Clear the sender's transmit buffer.
            with sender._tx_buffer.get_lock():
                for j in range(tx_data_len + 2):
                    sender._tx_buffer[j] = 0

                sender.tx_index.value = 0

        # Make sure that the transceivers are stopped.
        for xcvr in xvcrs.values():
            xcvr.stop()


# endregion

if __name__ == '__main__':
    main()
