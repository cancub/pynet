#!/usr/bin/env python
from __future__ import annotations

from math import ceil
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event, Lock
from unittest import TestCase, main
from unittest.mock import Mock, call, patch

from pynet.physical.base import CommsManager, ConnRequest, Medium, Transceiver
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
)
from pynet.testing import LogTestingMixin, ProcessBuilderMixin

BASE_TARGET = 'pynet.physical.base'

# region Helper Classes


class MockMedium(Medium, dimensionality=1, velocity_factor=0.77):
    """A helper class for testing the :class:`Medium` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _connect(self, *args, **kwargs) -> int:
        with self._receivers_lock:
            return len(self._receivers)


class MockTransceiver(Transceiver, supported_media=[MockMedium], buffer_bytes=1500):
    """A helper class for testing the :class:`Transceiver` class. Defines the necessary
    abstract methods to instantiate the class."""

    def _process_rx_value(self, data, *args, **kwargs):
        return data

    def _next_tx_symbol(self, data, *args, **kwargs):
        return data


# endregion

# region Tests


class TestCommsManager(TestCase):
    def setUp(self):
        super().setUp()
        self.cmgr = self._build_cmgr()

    def _build_cmgr(
        self, conn=None, buff_size=128, slot_ns=100, init_ns=0
    ) -> CommsManager:
        return CommsManager(
            conn=conn or Mock(), buff_size=buff_size, slot_ns=slot_ns, init_ns=init_ns
        )

    def test__init__(self):
        mock_conn = Mock()
        buff_size = 1024
        slot_ns = 100
        init_ns = 1000

        cmgr = CommsManager(
            conn=mock_conn, buff_size=buff_size, slot_ns=slot_ns, init_ns=init_ns
        )

        self.assertEqual(cmgr._conn, mock_conn)
        self.assertEqual(cmgr._buffer._data, [0] * buff_size)
        self.assertEqual(cmgr._slot_ns, slot_ns)
        self.assertEqual(cmgr.next_tx_ns, init_ns)
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
        orig_next_tx_ns = cmgr.next_tx_ns
        cmgr._conn.recv.return_value = (
            symbol := 1.5,
            duration_ns := 5,
            tx_delta_ns := 10,
        )

        self.assertEqual(cmgr.trigger_tx(), (symbol, duration_ns * TIME_DILATION_FACTOR))
        self.assertEqual(
            cmgr.next_tx_ns, orig_next_tx_ns + tx_delta_ns * TIME_DILATION_FACTOR
        )
        cmgr._conn.send.assert_called_once_with((CommsType.TRANSMIT, None))

    def test_trigger_rx(self):
        cmgr = self.cmgr
        conn = cmgr._conn
        orig_next_rx_ns = cmgr.next_rx_ns
        cmgr._buffer._data[0] = data = 1
        conn.recv.return_value = rx_delta_ns = 10

        cmgr.trigger_rx()
        self.assertEqual(
            cmgr.next_rx_ns, orig_next_rx_ns + rx_delta_ns * TIME_DILATION_FACTOR
        )
        conn.send.assert_called_once_with((CommsType.RECEIVE, data))

    def test_advance_rx_buffer(self):
        cmgr = self.cmgr

        cmgr._buffer._data = [1, 2, 3, 4, 5]

        cmgr.advance_rx_buffer()
        self.assertEqual(cmgr._buffer._data, [0, 2, 3, 4, 5])

        cmgr.advance_rx_buffer()
        self.assertEqual(cmgr._buffer._data, [0, 0, 3, 4, 5])

    def test_duration_in_slots(self):
        cmgr = self.cmgr
        slot_ns = cmgr._slot_ns

        self.assertEqual(cmgr._duration_in_slots(slot_ns), 1)
        self.assertEqual(cmgr._duration_in_slots(slot_ns - 1), 1)
        self.assertEqual(cmgr._duration_in_slots(slot_ns + 1), 2)

    def test_modify_buffer(self):
        slot_ns = 100
        cmgr = self._build_cmgr(buff_size=5, slot_ns=slot_ns)
        cmgr._buffer._data = [1, 2, 3, 4, 5]

        cmgr.modify_buffer(
            symbol=1.5, start_delay_ns=slot_ns * 1.1, duration_ns=slot_ns * 2.5
        )

        # Starting two slots after the current slot, and ending 3 slots thereafter
        self.assertEqual(cmgr._buffer._data, [1, 2, 4.5, 5.5, 6.5])

        # Move one slot into the future
        cmgr.advance_rx_buffer()
        self.assertEqual(cmgr._buffer._data, [0, 2, 4.5, 5.5, 6.5])

        # Test the wrapping around of data and also what happens when the start delay and
        # the duration of the transmission are exact multiples of the slot size.
        cmgr.modify_buffer(symbol=-0.5, start_delay_ns=slot_ns, duration_ns=slot_ns * 4)

        # The transmission occurs
        self.assertEqual(cmgr._buffer._data, [-0.5, 2, 4, 5, 6])


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
        diameter = 100
        max_baud = 1000
        v_ns = MockMedium._medium_velocity_ns

        for auto_start in (True, False):
            with self.subTest(auto_start=auto_start):
                medium = self.build_medium(
                    auto_start=auto_start, diameter=diameter, max_baud=max_baud
                )

                self.assertEqual(medium._dimensionality, MockMedium._dimensionality)
                self.assertEqual(medium._comms_details, {})
                self.assertEqual(medium._last_transition_ns, 0)
                self.assertEqual(medium._diameter, diameter)
                self.assertEqual(
                    (slot_ns := medium._slot_ns),
                    NANOSECONDS_PER_SECOND // (32 * max_baud),
                )
                self.assertEqual(medium._buffer_size, ceil((diameter / v_ns) / slot_ns))

                self.assertIsInstance(medium._comms_details_lock, Lock)
                self.assertIsInstance(medium._connection_queue, Queue)
                self.assertIsInstance(medium._stop_event, Event)

                if auto_start:
                    mock_start.assert_called_once()
                else:
                    mock_start.assert_not_called()

                self.assertEqual(Medium._instances[MockMedium, medium.name], medium)

            mock_start.reset_mock()

    def test__init__bad_diameter(self):
        for diameter in (-1, 0):
            with self.subTest(diameter=diameter):
                with self.assertRaisesRegex(ValueError, '`diameter` must be positive'):
                    self.build_medium(diameter=diameter)

    def test__init__bad_max_baud(self):
        for max_baud in (-1, 0):
            with self.subTest(max_baud=max_baud):
                with self.assertRaisesRegex(ValueError, '`max_baud` must be positive'):
                    self.build_medium(max_baud=max_baud)

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
    @patch(f'{BASE_TARGET}.monotonic_ns')
    @patch(f'{BASE_TARGET}.Thread', autospec=True)
    @patch.object(MockMedium, '_process_medium')
    def test_run(self, process_mock, ThreadMock, montonic_mock, *mocks):
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

        montonic_mock.assert_called_once()

        self.assertEqual(medium._last_transition_ns, montonic_mock.return_value)

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
    @patch.object(MockMedium, '_disconnect')
    @patch.object(MockMedium, '_connect')
    def test_monitor_connections_add_connection_success(
        self, connect_mock, disconnect_mock, CommsManagerMock, *mocks
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
    @patch.object(MockMedium, '_maybe_advance_rx_buffers')
    def test_process_medium(
        self, maybe_advance_rx_buffers_mock, monotonic_ns_mock, process_mock
    ):
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
        maybe_advance_rx_buffers_mock.assert_called_once_with(current_ns)
        process_mock.assert_has_calls(
            [call(loc1, cmgr1, current_ns), call(loc2, cmgr2, current_ns)]
        )

    def test_maybe_advance_rx_buffers(self):
        medium = self.build_medium()

        medium._last_transition_ns = orig_trans_ns = 100
        medium._slot_ns = slot_ns = 300

        with self.subTest('rx buffers not advanced'):
            medium._maybe_advance_rx_buffers(orig_trans_ns + slot_ns - 1)

            # Still in the same slot.
            self.assertEqual(medium._last_transition_ns, orig_trans_ns)

        loc1 = 123
        loc2 = 456

        cmgr1 = Mock()
        cmgr2 = Mock()

        medium._comms_details = {loc1: cmgr1, loc2: cmgr2}

        medium._last_transition_ns = trans_ns = 100
        medium._slot_ns = slot_ns = 300

        with self.subTest('rx buffers advanced'):
            with self.assertInTargetLogs(
                'DEBUG', f'{medium}: crossed slot boundary, advancing rx buffers'
            ):
                medium._maybe_advance_rx_buffers(trans_ns + slot_ns)

            self.assertEqual(medium._last_transition_ns, trans_ns + slot_ns)

            for cmgr in (cmgr1, cmgr2):
                cmgr.advance_rx_buffer.assert_called_once()

    @patch.object(MockMedium, '_calculate_travel_time_ns')
    def test_process_tx_rx_events(self, calc_travel_time_mock):
        medium = self.build_medium()

        current_ns = 500
        loc = 123
        cmgr = Mock(next_rx_ns=current_ns - 1, next_tx_ns=current_ns + 1)

        cmgr.trigger_rx.side_effect = ConnectionError(err_str := 'test')

        with self.subTest('rx triggered with error'):

            with self.assertInTargetLogs(
                'ERROR', f'{medium}: error while triggering rx: {err_str}'
            ):
                medium._process_tx_rx_events(loc, cmgr, current_ns)

            cmgr.trigger_rx.assert_called_once()
            cmgr.trigger_tx.assert_not_called()

        cmgr.reset_mock()
        cmgr.trigger_rx.side_effect = None

        with self.subTest('rx triggered, tx not triggered'):
            with self.assertInTargetLogs(
                'DEBUG', f'{medium}: triggering rx at location={loc}'
            ):
                medium._process_tx_rx_events(loc, cmgr, current_ns)

            cmgr.trigger_rx.assert_called_once()
            cmgr.trigger_tx.assert_not_called()

        cmgr.reset_mock()

        medium._last_transition_ns = last_ns = 100

        current_ns = 500
        loc = 123
        symbol = 0.5
        ns_per_symbol = 100
        next_tx_ns = current_ns - 1
        cmgr.next_rx_ns = current_ns + 1
        cmgr.next_tx_ns = next_tx_ns

        tx_offset_ns = next_tx_ns - last_ns

        dest_loc = 456
        dest_cmgr = Mock()
        medium._comms_details = {dest_loc: dest_cmgr, loc: cmgr}

        cmgr.trigger_tx.return_value = symbol, ns_per_symbol

        with self.subTest('rx not triggered, tx triggered'):
            with self.assertInTargetLogs(
                'DEBUG',
                [
                    f'{medium}: transceiver at location={loc} transmitting {symbol=}',
                    f'{medium}: adding {symbol=} to buffer for location={dest_loc}',
                ],
            ):
                medium._process_tx_rx_events(loc, cmgr, current_ns)

            cmgr.trigger_rx.assert_not_called()
            cmgr.trigger_tx.assert_called_once()

            calc_travel_time_mock.assert_called_once_with(loc, dest_loc)

            # Also make sure that the destination connection manager's modify_buffer method
            # has been called (but not the source's).
            dest_cmgr.modify_buffer.assert_called_once_with(
                symbol, tx_offset_ns + calc_travel_time_mock.return_value, ns_per_symbol
            )
            cmgr.modify_butter.assert_not_called()

        cmgr.reset_mock()

        cmgr.trigger_tx.side_effect = ConnectionError(err_str := 'test')

        with self.subTest('tx triggered with error'):
            with self.assertInTargetLogs(
                'ERROR', f'{medium}: error while triggering tx: {err_str}'
            ):
                medium._process_tx_rx_events(loc, cmgr, current_ns)

            cmgr.trigger_rx.assert_not_called()
            cmgr.trigger_tx.assert_called_once()

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
        Simulate a transmission triggered in a source in the right time slot and received
        by a destination in the right time slot.
        """
        medium = self.build_medium()
        buff_size = medium._buffer_size
        slot_ns = medium._slot_ns
        empty_buffer = [0] * buff_size
        symbol = 0.3
        duration = 10 * slot_ns
        duration_in_slots = ceil(duration / slot_ns)

        # The details about the transceivers.
        src_loc = 100
        dest_loc = 105
        src_conn = Mock()
        dest_conn = Mock()

        travel_time_ns = medium._calculate_travel_time_ns(src_loc, dest_loc)

        # Configure the connection managers such that all tx and rx events start at the
        # beginning of the next slot except for the source transceiver's transmission,
        # which will occur during the next run of _process_medium.
        src_cmgr = CommsManager(src_conn, buff_size, slot_ns, slot_ns)
        dest_cmgr = CommsManager(dest_conn, buff_size, slot_ns, slot_ns)

        monotonic_mock.return_value = current_time = slot_ns // 2
        src_cmgr.next_tx_ns = src_tx_time = current_time - 1
        dest_start_delay_slots = ceil((travel_time_ns + src_tx_time) / slot_ns)

        # Configure the medium with the managers.
        medium._comms_details = {src_loc: src_cmgr, dest_loc: dest_cmgr}

        # Prepare the transmission.
        src_conn.recv.return_value = (symbol, duration, slot_ns)

        # Process the medium and confirm that the symbol was placed in the buffer of
        # only the destination transceiver and with the correct slot offset and duration.
        medium._process_medium()

        src_conn.send.assert_called_once_with((CommsType.TRANSMIT, None))
        src_conn.recv.assert_called_once()
        dest_conn.send.assert_not_called()
        dest_conn.recv.assert_not_called()

        # The confirm that the receive buffer of the source is still pristine.
        self.assertEqual(src_cmgr._buffer._data, empty_buffer)

        # Meanwhile, the buffer for the destination has the symbol at the correct start
        # slot and with the correct slot duration.
        remainder_slots = buff_size - dest_start_delay_slots - duration_in_slots
        self.assertEqual(
            dest_cmgr._buffer._data,
            (
                [0] * dest_start_delay_slots
                + [symbol] * duration_in_slots
                + [0] * remainder_slots
            ),
        )

        src_conn.reset_mock()
        dest_conn.reset_mock()

        # Walk through the slots right up until the one containing the beginning of the
        # transmission reception at the destiantion and confirm that all transmissions and
        # receptions are zeros.
        for _ in range(dest_start_delay_slots - 1):
            # Neither the source nor the destination will transmit during this time, but
            # they will still be polled for their transmission and reception details.
            src_conn.recv.side_effect = [slot_ns, (0, duration, slot_ns)]
            dest_conn.recv.side_effect = [slot_ns, (0, duration, slot_ns)]

            current_time += slot_ns
            monotonic_mock.return_value = current_time

            # Do the thing.
            medium._process_medium()

            for conn in (src_conn, dest_conn):
                # Medium -> transceiver: one reception of a zero and one ping to transmit.
                conn.send.assert_has_calls(
                    [call((CommsType.RECEIVE, 0)), call((CommsType.TRANSMIT, None))]
                )

                # Transceiver -> medium: one reply with the next rx delta, one reply with
                # the next symbol, duration and tx delta.
                self.assertEqual(2, conn.recv.call_count)

            src_conn.reset_mock()
            dest_conn.reset_mock()

        # Walk through the slots which include the transmission on the link and confirm
        # that the receiver is receiving the correct symbol.
        for _ in range(duration_in_slots):
            # Neither the source nor the destination will transmit during this time, but
            # they will still be polled for their transmission and reception details.
            src_conn.recv.side_effect = [slot_ns, (0, duration, slot_ns)]
            dest_conn.recv.side_effect = [slot_ns, (0, duration, slot_ns)]

            current_time += slot_ns
            monotonic_mock.return_value = current_time

            # Do the thing.
            medium._process_medium()

            src_conn.send.assert_has_calls(
                [
                    call((CommsType.RECEIVE, 0)),
                    call((CommsType.TRANSMIT, None)),
                ]
            )
            self.assertEqual(2, src_conn.recv.call_count)

            dest_conn.send.assert_has_calls(
                [
                    call((CommsType.RECEIVE, symbol)),
                    call((CommsType.TRANSMIT, None)),
                ]
            )
            self.assertEqual(2, dest_conn.recv.call_count)

            src_conn.reset_mock()
            dest_conn.reset_mock()

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

                def _process_rx_value(self, symbol):
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

    @patch.object(MockTransceiver, '_process_rx_value')
    def test_process_reception_event(self, process_mock):
        xcvr = self.build_xcvr()
        conn = self.build_unique_mock('send')
        value = 1.3

        process_mock.side_effect = ValueError(err_str := 'bad process')

        with self.subTest('Error raised'):
            with self.assertInTargetLogs(
                'ERROR',
                f'{xcvr}: Error processing {value=}: {err_str}',
            ):
                xcvr._process_reception_event(conn, value)

            conn.send.assert_called_once_with(xcvr._base_delta_ns)

            process_mock.assert_called_once_with(value)

        process_mock.reset_mock()
        conn.send.reset_mock()
        process_mock.side_effect = None
        process_mock.return_value = delta = 123

        with self.subTest('No error raised'):
            xcvr._process_reception_event(conn, value)

            conn.send.assert_called_with(delta)

            process_mock.assert_called_once_with(value)

    @patch.object(MockTransceiver, '_next_tx_symbol')
    def test_process_transmission_event(self, next_mock):
        xcvr = self.build_xcvr()
        conn = self.build_unique_mock('send')

        next_mock.side_effect = ValueError(next_err_str := 'bad next')
        conn.send.side_effect = ValueError(send_err_str := 'bad send')

        with self.subTest('Errors raised'):
            with self.assertInTargetLogs(
                'ERROR',
                (
                    f'{xcvr}: Error getting next symbol to transmit: {next_err_str}',
                    f'{xcvr}: Error sending symbol=0 to medium: {send_err_str}',
                ),
            ):
                xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            conn.send.assert_called_once_with(0, xcvr._base_delta_ns, xcvr._base_delta_ns)

        next_mock.reset_mock()
        conn.send.reset_mock()
        next_mock.side_effect = None
        next_mock.return_value = symbol = 1

        with self.subTest('No errors raised'):
            xcvr._process_transmission_event(conn)

            next_mock.assert_called_once()
            conn.send.assert_called_once_with(
                symbol, xcvr._base_delta_ns, xcvr._base_delta_ns
            )

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
