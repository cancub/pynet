#!/usr/bin/env python
from __future__ import annotations

from math import sin
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event, Lock
from unittest import TestCase, main
from unittest.mock import Mock, call, patch

from pynet.physical.base import (
    CommsManager,
    ConnRequest,
    Medium,
    Transceiver,
    Transmission,
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

    def _process_rx_amplitude(self, data, *args, **kwargs):
        return data

    def _next_tx_symbol(self, data, *args, **kwargs):
        return data


# endregion

# region Tests


class TestTransmission(TestCase):
    def test_read(self):
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
                tx.get_amplitude(start_ns + duration_ns)


class TestCommsManager(TestCase, LogTestingMixin):
    log_target: str = BASE_TARGET

    def setUp(self):
        super().setUp()
        self.cmgr = self._build_cmgr()

    def _build_cmgr(self, conn=None, init_ns=0) -> CommsManager:
        return CommsManager(conn=conn or Mock(), init_ns=init_ns)

    def test__init__(self):
        mock_conn = Mock()
        init_ns = 1000

        cmgr = CommsManager(conn=mock_conn, init_ns=init_ns)

        self.assertEqual(cmgr._conn, mock_conn)
        self.assertEqual(cmgr.next_rx_ns, init_ns)

    def test_send(self):
        cmgr = self.cmgr
        conn = cmgr._conn
        data = 1

        conn.send.side_effect = ConnectionError(err_str := 'test')

        with self.subTest('ConnectionError'):
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

        conn.recv.side_effect = ConnectionError(err_str := 'test')

        with self.subTest('ConnectionError'):
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

    def test_trigger_rx(self):
        cmgr = self.cmgr
        conn = cmgr._conn
        orig_next_rx_ns = cmgr.next_rx_ns
        conn.recv.return_value = rx_delta_ns = 10
        dilated_rx_delta_ns = rx_delta_ns * TIME_DILATION_FACTOR

        with self.subTest('No ongoing transmissions'):
            cmgr.trigger_rx()
            self.assertEqual(cmgr.next_rx_ns, orig_next_rx_ns + dilated_rx_delta_ns)
            conn.send.assert_called_once_with((CommsType.RECEIVE, 0.0))

        orig_next_rx_ns = cmgr.next_rx_ns

        with self.subTest('Two ongoing transmissions plus one completed and one error'):
            tx1 = Mock()
            tx2 = Mock()
            tx3 = Mock()
            tx4 = Mock()

            tx1.get_amplitude.return_value = amp1 = 1.2
            tx2.get_amplitude.return_value = amp2 = -0.5
            tx3.get_amplitude.side_effect = TransmissionComplete
            tx4.get_amplitude.side_effect = ValueError(err_str := 'test')

            cmgr._transmissions = {tx1, tx2, tx3, tx4}

            with self.assertInTargetLogs(
                'ERROR',
                f'{cmgr}: an exception occurred while processing a transmission: {err_str}',
            ):
                cmgr.trigger_rx()

            self.assertEqual(cmgr.next_rx_ns, orig_next_rx_ns + dilated_rx_delta_ns)
            conn.send.assert_called_with((CommsType.RECEIVE, amp1 + amp2))

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

    # region _monitor_connections

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
        medium._connection_queue.get.return_value = creq = ConnRequest(
            mock_conn, None, True, kwargs
        )

        # Make the connection fail.
        err_str = 'Boom!'
        connect_mock.side_effect = err = Exception(err_str)

        # Use this test to check the debug logs as well.
        with self.assertInTargetLogs('ERROR', f'{medium}: connection failed: {err_str}'):
            medium._monitor_connections()

        # Confirm that there was one connection and no disconnection.
        connect_mock.assert_called_once_with(creq)
        mock_conn.send.assert_called_once_with((Responses.ERROR, err))
        disconnect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch.object(MockMedium, '_disconnect')
    @patch.object(MockMedium, '_connect')
    def test_monitor_connections_add_connection_reply_fails(
        self, connect_mock, disconnect_mock, *mocks
    ):
        medium = self.build_medium()
        mock_conn = Mock()
        kwargs = {'foo': 'bar'}

        # Allow for one loop to perform a connection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the new connection.
        medium._connection_queue.get.return_value = creq = ConnRequest(
            mock_conn, None, True, kwargs
        )
        location = connect_mock.return_value

        # Make the reply fail.
        mock_conn.send.side_effect = Exception(err_str := 'Boom!')

        # Use this test to check the debug logs as well.
        with self.assertInTargetLogs(
            'ERROR', f'{medium}: connection info reply to transceiver failed: {err_str}'
        ):
            medium._monitor_connections()

        # Confirm that there was one connection and no disconnection.
        connect_mock.assert_called_once_with(creq)
        mock_conn.send.assert_called_once_with((Responses.OK, location))
        disconnect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    @patch(f'{BASE_TARGET}.CommsManager', autospec=True)
    @patch(f'{BASE_TARGET}.monotonic_ns', autospec=True)
    @patch.object(MockMedium, '_disconnect')
    @patch.object(MockMedium, '_connect')
    def test_monitor_connections_add_connection_success(
        self, connect_mock, disconnect_mock, mono_mock, CommsManagerMock, *mocks
    ):
        medium = self.build_medium()
        mock_conn = Mock()
        kwargs = {'foo': 'bar'}

        # Allow for one loop to perform a connection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the new connection.
        medium._connection_queue.get.return_value = creq = ConnRequest(
            mock_conn, None, True, kwargs
        )
        location = connect_mock.return_value
        cmgr = CommsManagerMock.return_value

        # Use this test to check the debug logs as well.
        with self.assertInTargetLogs(
            'DEBUG',
            ['Starting connection worker thread', 'Connection worker thread exiting'],
        ):
            medium._monitor_connections()

        # Confirm that there was one connection and no disconnection.
        connect_mock.assert_called_once_with(creq)
        CommsManagerMock.assert_called_once_with(mock_conn, mono_mock.return_value)
        mock_conn.send.assert_called_once_with((Responses.OK, location))
        self.assertEqual(medium._comms_details[location], cmgr)

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

        # Allow for one loop to perform a disconnection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the existing connection.
        medium._connection_queue.get.return_value = creq = ConnRequest(
            None, location, False, None
        )
        medium._comms_details[location] = cmgr = Mock()

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
            medium._monitor_connections()

        # Confirm that there was one disconnection and no connection.
        disconnect_mock.assert_called_once_with(creq)
        cmgr.conn.close.assert_called_once()
        self.assertEqual(medium._comms_details, {})

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

        # Allow for one loop to perform a disconnection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        # Set up the existing connection.
        medium._connection_queue.get.return_value = creq = ConnRequest(
            None, location, False, None
        )
        medium._comms_details[location] = cmgr = Mock()

        medium._monitor_connections()

        # Confirm that there was one disconnection and no connection.
        disconnect_mock.assert_called_once_with(creq)
        cmgr.conn.close.assert_called_once()
        self.assertEqual(medium._comms_details, {})

        connect_mock.assert_not_called()

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch(f'{BASE_TARGET}.Queue', autospec=True)
    def test_monitor_connections_got_bad_connection_request(self, *mocks):
        medium = self.build_medium()
        medium._stop_event.is_set.return_value = True

        # Allow for one loop to perform a connection and then exit.
        medium._stop_event.is_set.side_effect = [False, True]

        medium._connection_queue.get.return_value = 123

        with self.assertInTargetLogs(
            'ERROR',
            f'{medium}: unexpected connection details type: int',
        ):
            medium._monitor_connections()

    # endregion

    # region _process_medium-related

    @patch.object(MockMedium, '_process_tx_rx_events')
    @patch(f'{BASE_TARGET}.monotonic_ns')
    def test_process_medium(self, monotonic_ns_mock, process_mock):
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
        process_mock.assert_has_calls(
            [call(loc1, cmgr1, current_ns), call(loc2, cmgr2, current_ns)]
        )

    @patch.object(MockMedium, '_calculate_travel_time_ns')
    def test_process_tx_rx_events(self, calc_travel_time_mock):
        medium = self.build_medium()

        symbol_fn = Mock()
        current_ns = 500
        duration_ns = 100

        src_loc = 123
        src_cmgr = Mock(next_rx_ns=current_ns - 1)

        dest_loc = 456
        dest_cmgr = Mock()

        src_trigger_rx = src_cmgr.trigger_rx
        src_trigger_tx = src_cmgr.trigger_tx
        src_add_tranmsission = src_cmgr.add_transmission
        dest_add_tranmsission = dest_cmgr.add_transmission

        def reset_mocks():
            src_cmgr.reset_mock()
            src_trigger_rx.reset_mock()
            src_trigger_tx.reset_mock()
            calc_travel_time_mock.reset_mock()
            src_add_tranmsission.reset_mock()
            dest_add_tranmsission.reset_mock()

            src_trigger_rx.side_effect = None
            src_trigger_tx.side_effect = None

        src_trigger_rx.side_effect = ConnectionError(err_str := 'test')
        src_trigger_tx.return_value = None

        with self.subTest('rx triggered with error, tx triggered with no symbol'):

            with self.assertInTargetLogs(
                'ERROR', f'{medium}: error while triggering rx: {err_str}'
            ):
                medium._process_tx_rx_events(src_loc, src_cmgr, current_ns)

            src_trigger_rx.assert_called_once()
            src_trigger_tx.assert_called_once()

            src_add_tranmsission.assert_not_called()
            dest_add_tranmsission.assert_not_called()

        reset_mocks()

        medium._comms_details = {dest_loc: dest_cmgr, src_loc: src_cmgr}

        with self.subTest('rx not triggered, tx triggered for first symbol'):
            # No ongoing tranmissions
            medium._transmission_completion_times = {}

            src_cmgr.next_rx_ns = current_ns + 1

            src_trigger_tx.return_value = symbol_fn, duration_ns
            calc_travel_time_mock.return_value = prop_delay_ns = 123

            with self.assertInTargetLogs(
                'DEBUG',
                f'{medium}: transceiver at location={src_loc} beginning transmission of new symbol',
            ):
                medium._process_tx_rx_events(src_loc, src_cmgr, current_ns)

            src_trigger_rx.assert_not_called()
            src_trigger_tx.assert_called_once()

            self.assertEqual(
                medium._transmission_completion_times,
                {src_cmgr: current_ns + duration_ns},
            )

            calc_travel_time_mock.assert_called_once_with(src_loc, dest_loc)

            dest_add_tranmsission.assert_called_once_with(
                symbol_fn, current_ns + prop_delay_ns, duration_ns, 1
            )
            src_add_tranmsission.assert_not_called()

        reset_mocks()

        with self.subTest('rx not triggered, tx triggered for next symbol'):
            # A transmission that recently completed.
            stop_ns = current_ns - 30
            medium._transmission_completion_times = {src_cmgr: stop_ns}

            with self.assertInTargetLogs(
                'DEBUG',
                f'{medium}: transceiver at location={src_loc} beginning transmission of new symbol',
            ):
                medium._process_tx_rx_events(src_loc, src_cmgr, current_ns)

            src_trigger_rx.assert_not_called()
            src_trigger_tx.assert_called_once()

            self.assertEqual(
                medium._transmission_completion_times, {src_cmgr: stop_ns + duration_ns}
            )

            calc_travel_time_mock.assert_called_once_with(src_loc, dest_loc)

            # Also make sure that the destination connection manager has received the
            # symbol (and that the source has not).
            dest_add_tranmsission.assert_called_once_with(
                symbol_fn, stop_ns + prop_delay_ns, duration_ns, 1
            )
            src_add_tranmsission.assert_not_called()

        reset_mocks()

        with self.subTest('rx not triggered, tx ongoing'):
            current_ns = stop_ns - 1

            medium._process_tx_rx_events(src_loc, src_cmgr, current_ns)

            src_trigger_rx.assert_not_called()
            src_trigger_tx.assert_not_called()

            calc_travel_time_mock.assert_not_called()

            dest_add_tranmsission.assert_not_called()
            src_add_tranmsission.assert_not_called()

        reset_mocks()

        with self.subTest('tx triggered with error'):
            current_ns = stop_ns + duration_ns + 1

            # Ignore receptions.
            src_cmgr.next_rx_ns = current_ns + 1
            dest_cmgr.next_rx_ns = current_ns + 1

            src_trigger_tx.side_effect = ConnectionError(err_str := 'test')

            with self.assertInTargetLogs(
                'ERROR', f'{medium}: error while triggering tx: {err_str}'
            ):
                medium._process_tx_rx_events(src_loc, src_cmgr, current_ns)

            src_trigger_rx.assert_not_called()
            src_trigger_tx.assert_called_once()

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
        init_ns = 100
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
        stop_ns = start_ns + duration_ns

        # Configure the connection managers such that all rx events are upcoming.
        src_cmgr = CommsManager(src_conn, init_ns)
        dest_cmgr = CommsManager(dest_conn, init_ns)
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
            # recv() should return the next reception time, and the second should return
            # None for no transmission
            dest_recv.side_effect = [dest_next_rx_delta_ns, None]

            medium._process_medium()

            # No signal has reached the destination yet.
            dest_send.assert_has_calls(
                [call((CommsType.RECEIVE, 0.0)), call((CommsType.TRANSMIT, None))]
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
            # recv() should return the next reception time, and the second should return
            # None for no transmission
            dest_recv.side_effect = [dest_next_rx_delta_ns, None]

            medium._process_medium()

            # The destination should have begun receiving the symbol.
            amplitude = symbol_fn((orig_dest_rx - start_ns) / TIME_DILATION_FACTOR)
            dest_send.assert_has_calls(
                [call((CommsType.RECEIVE, amplitude)), call((CommsType.TRANSMIT, None))]
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
            # recv() should return the next reception time, and the second should return
            # None for no transmission
            dest_recv.side_effect = [dest_next_rx_delta_ns, None]

            medium._process_medium()

            dest_send.assert_has_calls(
                [call((CommsType.RECEIVE, 0.0)), call((CommsType.TRANSMIT, None))]
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
            with self.assertRaises(TypeError):

                class BadTransceiver1(Transceiver, supported_media=[object]):
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
            self.assertEqual(list(GoodTransceiver._tx_buffer), [0] * buffer_bytes)
            self.assertEqual(list(GoodTransceiver._rx_buffer), [0] * buffer_bytes)

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
    def test_run_connect_disconnect(self, PipeMock, ThreadMock, EventMock):
        mock_thread = ThreadMock.return_value

        PipeMock.side_effect = [
            (conn_listener_mock := Mock(), None),
            (None, None),
        ]
        EventMock.side_effect = [
            proc_stop_event := self.build_unique_mock('is_set'),
            thread_stop_event := Mock(),
        ]

        proc_stop_event.is_set.side_effect = [False, False, True]

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
        thread_stop_event.clear.assert_called_once()
        ThreadMock.assert_called_once_with(
            target=xcvr._monitor_medium,
            args=(conn,),
        )
        mock_thread.start.assert_called_once()

        # Disconnect
        thread_stop_event.set.assert_called_once()
        conn.send.assert_called_once_with(ManagementMessages.CLOSE)
        mock_thread.join.assert_called_once()

    @patch.object(MockTransceiver, '_process_rx_amplitude')
    def test_process_reception_event(self, process_mock):
        xcvr = self.build_xcvr()
        conn = self.build_unique_mock('send')
        amplitude = 1.3

        process_mock.side_effect = ValueError(err_str := 'ruh roh')

        with self.subTest('Error raised'):
            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error processing {amplitude=}: {err_str}',
            ):
                xcvr._process_reception_event(conn, amplitude)

            conn.send.assert_called_once_with(xcvr._base_delta_ns)

            process_mock.assert_called_once_with(amplitude)

        process_mock.reset_mock()
        conn.send.reset_mock()
        process_mock.side_effect = None
        process_mock.return_value = delta = 123

        with self.subTest('No error raised'):
            xcvr._process_reception_event(conn, amplitude)

            conn.send.assert_called_with(delta)

            process_mock.assert_called_once_with(amplitude)

    @patch.object(MockTransceiver, '_next_tx_symbol')
    def test_process_transmission_event(self, next_mock):
        xcvr = self.build_xcvr()
        conn = self.build_unique_mock('send')
        send = conn.send
        base_delta_ns = xcvr._base_delta_ns

        next_mock.side_effect = ValueError(err_str := 'bad next')

        with self.subTest('Failed to get next symbol'):
            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error getting next symbol to transmit: {err_str}',
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once_with(None)

        next_mock.reset_mock()
        send.reset_mock()
        next_mock.side_effect = None
        next_mock.return_value = symbol_fn = Mock(
            side_effect=ValueError(err_str := 'bad symbol')
        )

        with self.subTest('Bad symbol function'):
            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error getting amplitude from symbol function: {err_str}',
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            symbol_fn.assert_called_once_with(0)
            send.assert_called_once_with(None)

        next_mock.reset_mock()
        send.reset_mock()
        amplitude = 'hi'
        next_mock.return_value = symbol_fn = Mock(return_value=amplitude)

        with self.subTest('Good symbol function, bad return type'):
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

        next_mock.reset_mock()
        send.reset_mock()
        value = 1.3
        next_mock.return_value = symbol_fn = Mock(return_value=value)

        with self.subTest('Good symbol function, good return type'):
            xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            symbol_fn.assert_called_once_with(0)
            send.assert_called_once_with(symbol_fn, base_delta_ns)

        next_mock.reset_mock()
        send.reset_mock()
        next_mock.return_value = symbol = 'hi'

        with self.subTest('Bad non-function'):
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

        next_mock.reset_mock()
        send.reset_mock()
        next_mock.return_value = symbol = 1.3

        with self.subTest('Good non-function'):
            xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once()

            fn, delta = send.call_args[0]

            self.assertEqual(delta, base_delta_ns)
            self.assertEqual(fn(0), symbol)

        next_mock.reset_mock()
        send.reset_mock()
        next_mock.return_value = symbol_fn = Mock(return_value=1)
        conn.send.side_effect = ValueError(err_str := 'bad send')

        with self.subTest('Error raised during send'):
            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error sending symbol function to medium: {err_str}',
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            send.assert_called_once_with(symbol_fn, base_delta_ns)

    @patch(f'{BASE_TARGET}.Event', autospec=True)
    @patch.object(MockTransceiver, '_process_reception_event')
    @patch.object(MockTransceiver, '_process_transmission_event')
    def test_monitor_medium(self, tx_mock, rx_mock, EventMock):
        stop_event_mock = EventMock.return_value
        xcvr = self.build_xcvr()

        conn = self.build_unique_mock('recv')

        stop_event_mock.is_set.side_effect = [False, True]

        conn.recv.side_effect = ValueError(err_str := 'bad recv')

        with self.subTest('Error raised during initial receive'):
            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error receiving data from medium: {err_str}',
            ):
                xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

        conn.recv.reset_mock()
        conn.close.reset_mock()
        stop_event_mock.is_set.reset_mock()
        conn.recv.side_effect = None

        # A second call will throw an exception.
        stop_event_mock.is_set.side_effect = [False]
        conn.recv.return_value = ManagementMessages.CLOSE

        with self.subTest('Close message received'):
            xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 1)

        conn.recv.reset_mock()
        conn.close.reset_mock()
        stop_event_mock.is_set.reset_mock()

        data = 123

        conn.recv.return_value = data
        stop_event_mock.is_set.side_effect = [False, True]

        with self.subTest('Bad data sent from medium'):
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

        conn.recv.reset_mock()
        conn.close.reset_mock()
        stop_event_mock.is_set.reset_mock()

        conn.recv.return_value = CommsType.RECEIVE, data
        stop_event_mock.is_set.side_effect = [False, True]

        with self.subTest('Receive event'):
            xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

            rx_mock.assert_called_once_with(conn, data)
            tx_mock.assert_not_called()

        conn.recv.reset_mock()
        conn.close.reset_mock()
        stop_event_mock.is_set.reset_mock()
        rx_mock.reset_mock()

        conn.recv.return_value = CommsType.TRANSMIT, data
        stop_event_mock.is_set.side_effect = [False, True]

        with self.subTest('Transmit event'):
            xcvr._monitor_medium(conn)

            conn.recv.assert_called_once()
            conn.close.assert_called_once()

            self.assertEqual(stop_event_mock.is_set.call_count, 2)

            tx_mock.assert_called_once_with(conn)
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
        base_baud = 1e6
        xvcrs = {n: MockTransceiver(name=n.title(), base_baud=base_baud) for n in names}

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
