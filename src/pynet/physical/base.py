"""
Base classes for physical media and transceivers, with these classes handling the
low-level details of connecting and disconnecting transceivers from media. Subclasses
of these base classes define the specifics of how transceivers connect to media and
how data is transmitted over the media.

:author: Alf O'Kenney

TODO:
- allow for a broad range of transmissions by refactoring the transmission of the
  Transceiver to accept
    - the duration of the transmission of a symbol
    - one of
        - a constant value (int or float)
        - a function whose input is the time since the beginning of the transmission of
            the symbol and whose output is the value of the signal at that time
- allow users to specify the clock frequency of the transceivers
- allow users to specify a PLL for transceivers
- implement subclass abstract base classes for electrical and optical media
- maybe further divide them into 1D and 3D media
- configure logging (esp. for processes)
- allow users to define the environment (e.g., temperature, humidity, walls, buildings,
    etc.) in which the transceivers and media are located
        - this will have on impact on what, if anything is received by devices
"""

from __future__ import annotations

import atexit
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from math import ceil
from multiprocessing import Array, Event, Lock, Pipe, Process, Queue, Value
from threading import Thread
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, Sequence
from weakref import WeakValueDictionary

from .constants import (
    DEFAULT_NAME,
    NANOSECONDS_PER_SECOND,
    SPEED_OF_LIGHT,
    TIME_DILATION_FACTOR,
    ManagementMessages,
    CommsType,
    Responses,
)
from .exceptions import ConnectionError, ProcessNotRunningError
from .utils import RingBuffer
from ..space import euclidean_distance

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from ..space import Location

__all__ = ['Medium', 'Transceiver']

log = logging.getLogger(__name__)


ConnRequest = namedtuple('ConnRequest', ['conn', 'location', 'create', 'kwargs'])


class CommsManager:
    """A container for managing the relationship between a :class:`Transceiver` and a
    :class:`Medium` which the latter uses to trigger all communications (i.e., both TX and
    RX) with the former.
    """

    def __init__(self, conn: Connection, buff_size: int, slot_ns: int, init_ns: int):
        """
        :param conn: The :class:`Medium`'s :class:`.Connection` of the :class:`.Pipe`
            between itself and a :class:`Transceiver`.
        :param buff_size: The size of the rx buffer that will store the signal to be
            received by a :class:`Transceiver`.
        :param slot_ns: The duration in nanoseconds of each sample in the quantized
            :class:`Medium`.
        :param init_ns: The initial time in nanoseconds at which a :class:`Transceiver`
            is to begin transmitting and receiving.
        """
        self._conn: Connection = conn

        # Create the buffer for the symbols to be received by the transceiver.
        self._buffer: RingBuffer = RingBuffer(size=buff_size, default=0)

        self._slot_ns: int = slot_ns

        self.next_tx_ns = init_ns
        self.next_rx_ns = init_ns

    def _send(self, data: Any) -> None:
        """Send data to the :class:`Transceiver`.

        :param data: The data to send to the :class:`Transceiver`.
        """
        try:
            self._conn.send(data)
        except Exception as e:
            raise ConnectionError(
                f'Could not send data to the Transceiver via {self}: {e}'
            )

    def _recv(self) -> Any:
        """Receive data from the :class:`Transceiver`.

        :returns: The data received from the :class:`Transceiver`.

        TODO:
        Add a timeout
        """
        try:
            return self._conn.recv()
        except Exception as e:
            raise ConnectionError(
                f'Could not receive data from the Transceiver via {self}: {e}'
            )

    def trigger_tx(self) -> tuple[float, float]:
        """Ping the :class:`Transceiver` to transmit a symbol and wait for a response
        containing the symbol to be transmitted and the symbol rate (in symbols per
        nanosecond) at which the symbol will be transmitted.

        :returns: The symbol to be transmitted and the duration in nanoseconds for the
            symbol to be put on the link.

        NOTE:
        The actual duration of the symbol on the link is time-dilated in order to allow
        the simulation to not trip over itself.

        TODO:
        gracefull recover from an exception by using a 0 symbol and the previous values
        for rate and delta. This will likely result in a loss of data, but it will prevent
        the simulation from crashing.
        """
        self._send((CommsType.TRANSMIT, None))

        # The transceiver will respond with three pieces of information:
        #   1. the symbol to be transmitted
        #   2. the symbol duration in nanoseconds
        #   3. the delta time to the next tx event
        symbol, duration_ns, tx_delta_ns = self._recv()

        # Update the next tx time.
        # NOTE:
        # The transceiver may no longer have any symbols to transmit, but we still
        # schedule it to transmit a symbol in the next time slot so that it keeps pace
        # with the medium. Technically a lack of a symbol to tansmit is the same as
        # transmitting a zero, so this is not a problem.
        self.next_tx_ns += tx_delta_ns * TIME_DILATION_FACTOR

        return symbol, duration_ns * TIME_DILATION_FACTOR

    def trigger_rx(self):
        """Send the next symbol in the rx buffer to the :class:`Transceiver`.

        TODO:
        Gracefully recover from an exception by using the previous delta time.
        """
        self._send((CommsType.RECEIVE, self._buffer.get_current()))

        # The transceiver will respond with the delta time to the next rx event.
        rx_delta_ns = self._recv()

        # Update the next rx time.
        self.next_rx_ns += rx_delta_ns * TIME_DILATION_FACTOR

    def advance_rx_buffer(self):
        """Used by the :class:`Transceiver` to push out oldest symbol from the rx buffer
        and add in a zero represnting dead air."""
        self._buffer.shift()

    def modify_buffer(self, symbol: int | float, start_delay_ns=float, duration_ns=float):
        """Used by the :class:`Medium` to modify certain slots of the rx buffer.

        :param symbol: The symbol to be added to the rx buffer.
        :param start_delay_ns: The time (in nanoseconds) after the start of the current
            slot that the wavefront appears at the :class:`Transceiver` represented by
            this :class:`CommsManager` object.
        :param duration_ns: The number of nanoseconds per symbol (the inverse of
            the symbol rate).
        """
        # The index in the rx buffer where we should begin adding symbol.
        start_index = self._duration_in_slots(start_delay_ns)

        # The number of times a symbol will transition from one time slot to another. In
        # other words, the number of times we see the symbol on the link during samples of
        # our quantized medium.
        slots = self._duration_in_slots(duration_ns)

        self._buffer.overlay([symbol] * slots, start_index)

    @lru_cache
    def _duration_in_slots(self, duration_ns=float) -> int:
        """Get the number of samples in the quantized :class:`Medium` that will occur in
        the given number of nanoseconds.

        TODO:
        A future optimization may be to just build a lookup table for this when the
        :class:`Medium` is initialized.

        :param duration_ns: The number of nanoseconds to be converted to time slots.
        :return: The number of time slots that the given number of nanoseconds.
        """
        return ceil(duration_ns / self._slot_ns)


class Medium(Process, ABC):
    """A base class for all physical media.

    Just as in real life, a :class:`Medium` is passive in its interactions with a
    :class:`Transceiver`. That is, a :class:`Medium` instance does not initiate any
    connections or disconnections, nor does it request data. In PyNet, it exists merely as
    a record-keeper of the :class:`Transceiver` instances that are connected to it and as
    a means of transporting data between those :class:`Transceiver` instances in a manner
    that simulates the physical medium.

    :class:`Medium` instances are forked processes that run in the background. This is in
    anticipation of larger, more complex simulations that will require significant counts
    of media and transceivers. In such simulations, low-level details such as clock
    synchronization and simulated delays become difficult to manage via either a single
    process or multiple threads in a single process (due to their interactions with the
    GIL). By running each :class:`Medium` instance in its own process, the hope is that we
    can mitigate these issues if not eliminate them altogether.

    Subclasses of :class:`Medium` define their own connection and disconnection logic by
    overriding the :meth:`_connect` and :meth:`_disconnect` methods. An example of such
    logic is confirming that there is enough space on the physical medium to accommodate a
    new connection (e.g., a coaxial cable to which 802.3 Ethernet transceivers connect via
    vampire taps).

    NOTE: Instances of :class:`Medium` subclasses must be interacted with via methods of
    a :class:`Transceiver` instance. That is, if a :class:`Medium` instance is to be
    connected to a :class:`Transceiver` instance, the :meth:`Transceiver.connect` method
    must be called with the desired :class:`Medium` instance as an argument.
    """

    # Keep track of all instances so we can shut down their respective processes at exit
    _instances = WeakValueDictionary()

    # The number of dimensions in which the medium exists. For example, coaxial cable is
    # 1D, while free space/air is 3D.
    _dimensionality: int

    # How fast (in meters per nanosecond) a signal travels along the medium. Related to
    # the velocity factor of the medium, which is the ratio of the speed of a wave
    # propagating along the medium to the speed of light.
    _medium_velocity_ns: float

    # region Dunders

    def __init_subclass__(cls, dimensionality: int, velocity_factor: float, **kwargs):
        """Initialize a subclass of :class:`Medium`.

        :param dimensionality: The number of dimensions in which the medium exists. For
            example, coaxial cable is 1D, while free space/air is 3D.
        :param max_baud: The maximum number of symbols per second that can be
            transmitted over the medium. Measurements are in symbols per second (baud).
        """
        super().__init_subclass__()
        if dimensionality not in (1, 3):
            raise ValueError('`dimensionality` must be 1 or 3')

        if not 0 < velocity_factor <= 1:
            raise ValueError('`velocity_factor` must be in range (0, 1]')

        cls._dimensionality = dimensionality
        cls._medium_velocity_ns = (
            velocity_factor * SPEED_OF_LIGHT / NANOSECONDS_PER_SECOND
        )

    def __new__(cls, name: str, *args, **kwargs):
        try:
            return cls._instances[cls, name]
        except KeyError:
            pass
        obj = super().__new__(cls)
        cls._instances[cls, name] = obj
        return obj

    def __init__(self, name: str, max_baud: int, diameter: int, auto_start: bool = True):
        """Initialize a :class:`Medium` instance.

        :param name: The name of the medium. This is used to identify the medium when
            connecting :class:`Transceiver` instances to it.
        :param max_baud: The maximum number of symbols per second that will be
            transmitted over the medium. Measurements are in symbols per second (baud).
        :param diameter: The maximum distance between devices using the medium.
            Measurements are in meters.
        :param auto_start: Whether or not to automatically start the process when the
            instance is created. Defaults to ``True``.
        """
        if max_baud <= 0:
            raise ValueError('`max_baud` must be positive')
        if diameter <= 0:
            raise ValueError('`diameter` must be positive')

        super().__init__()

        self.name = name

        self._connection_thread: Thread = None

        # The dict of details about connected transceiver, and the lock used to ensure
        # thread safety.
        self._comms_details: dict[Location, CommsManager] = {}
        self._comms_details_lock: Lock = Lock()

        # We must keep track of the last time we shifted buffers. This is to ensure that
        # we don't shift buffers more than once per symbol.
        self._last_transition_ns: int = 0

        # The Queue monitored by the worker thread for updates to state of connected
        # Transceivers.
        self._connection_queue: Queue = Queue()

        # This event is used to stop both ourselves (the forked Process) and Thread
        # workers.
        self._stop_event: Event = Event()

        # The maximum distance between devices using the medium. Measurements are in
        # meters.
        self._diameter: float = diameter

        # The first things we need to recognize is that we can't actually have a medium
        # object that manages continuous waveform. By constructing a medium with a program
        # that operates in discrete time steps, the best we can do is represent waveforms
        # as a series of discrete samples. This being the case, we need to consider how to
        # ensure that we accurately store any signal. Thankfully, we can reference the
        # Nyquist-Shannon sampling theorem, which states that we can avoid aliasing by
        # sampling at twice the maximum frequency of the waveform. In our case, it would
        # be nice to allow for simulation things like clock drift(?), where receivers will
        # need to adjust their sampling rate to match the sender's phase. So, we'll go
        # above and beyond the requirement to avoid aliasing and sample at 32 times the
        # maximum baud rate.
        # TODO:
        # Allow users to configure this?
        sample_rate = 32 * max_baud

        # Given this sample rate, how many seconds would there be between samples? In
        # other words, how many nanoseconds does each slot of a buffer represent?
        #   (ns / s) / (sample / s) = ns / sample
        self._slot_ns = NANOSECONDS_PER_SECOND // sample_rate

        # There's no need to store information any longer than we need to, and we can
        # determine the minimum buffer size by calculating the number of slots to required
        # to cover the maximum distance of the medium.
        # NOTE:
        # We can understand this as a sort of analog to measuring distance in light-years.
        # A light-year is unit of measurement that is equal to the distance that light
        # travels in one year. In our case, we're looking at the distance that a signal
        # in one slot in our medium and using that to measure the number of slots from one
        # end of the medium to the other.
        #   diameter_in_meters / meters_per_ns = ns_to_travese_medium
        #   ns_to_traverse_medium / ns_per_slot = number of slots to traverse medium
        self._buffer_size: int = (
            ceil((diameter / self._medium_velocity_ns) / self._slot_ns)
            * TIME_DILATION_FACTOR
        )

        # By default, we do the caller the service of starting up the Process rather than
        # making them remember to do it for each and every Medium instance.
        if auto_start:
            self.start()

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__}: name={self.name!r}, pid={self.pid}, '
            f'running={self.is_alive()}>'
        )

    # endregion

    # region Abstract methods

    @abstractmethod
    def _connect(self, **kwargs) -> Location:
        """Run class-specific connection logic

        NOTE:
        Part of the kwargs may be use to specify the desired location of the
        :class:`Transceiver` instance. In this case, this method should echo back the
        location on success.

        :kwargs: Any additional arguments required to connect to the medium.
        :returns: The location of the newly-connected Transceiver with respect to the
            Medium.
        """
        raise NotImplementedError

    # endregion

    # region Optional overrides

    def _disconnect(self):
        """Run class-specific disconnection logic"""
        pass

    # endregion

    # region Multi-processing- / IPC-related methods

    def run(self):
        log.info(f'Starting medium process ({self.pid})')

        # Start the worker thread to monitor connection changes.
        self._connection_thread = Thread(target=self._monitor_connections)
        self._connection_thread.start()

        # Start tracking the last time we shifted buffers.
        self._last_transition_ns = monotonic_ns()

        # Run the main loop.
        while not self._stop_event.is_set():
            self._process_medium()

        self._connection_thread.join()

        log.debug(f'{self}: shutting down')

    def _monitor_connections(self):
        """To be run in a separate thread, this method watches the connections queue for
        updates to the set of connected transceivers and connects or disconnects them as
        appropriate."""
        log.info(f'Starting connection worker thread ({self.pid})')

        while not self._stop_event.is_set():
            # Wait for a connection update.
            conn_req = self._connection_queue.get()
            if conn_req is ManagementMessages.CLOSE:
                break

            try:
                create = conn_req.create
            except AttributeError:
                cls_name = conn_req.__class__.__name__
                log.error(f'{self}: unexpected connection details type: {cls_name}')
                continue

            if create:
                self._add_connection(conn_req)
            else:
                self._remove_connection(conn_req)

        log.info(f'Connection worker thread exiting ({self.pid})')

    def _add_connection(self, conn_req: ConnRequest):
        """Add a connection between a :class:`Transceiver` and this :class:`Medium` and
        return the details of this connection to the :class:`Transceiver`.

        :param creq: The connection request to use to configure the connection.
        """
        conn = conn_req.conn

        try:
            location = self._connect(conn_req)
        except Exception as e:
            log.error(f'{self}: connection failed: {e}')
            conn.send((Responses.ERROR, e))
            return

        # Create a new CommsManager for this connection and configure it to begin sending
        # and receiving data in the next slot.
        new_cmgr = CommsManager(
            conn,
            self._buffer_size,
            self._slot_ns,
            self._last_transition_ns + self._slot_ns,
        )

        # Send the location back to the transceiver.
        try:
            conn.send((Responses.OK, location))
        except Exception as e:
            log.error(f'{self}: connection info reply to transceiver failed: {e}')

            # We cannot continue with this connection, since the transceiver is unaware of
            # it.
            return

        log.info(f'{self}: connection established at {location}')
        with self._comms_details_lock:
            self._comms_details[location] = new_cmgr

    def _remove_connection(self, conn_req: ConnRequest):
        """Remove a connection between a :class:`Transceiver` and this :class:`Medium`.

        :param creq: The connection request to use to locate the connection to remove.

        TODO:
        Perform some sanity checks to make sure that the connection is being closed by the
        correct transceiver.
        """
        try:
            self._disconnect(conn_req)
        except Exception as e:
            log.error(
                f'{self}: subclass disconnection failed: {e}. Continuing with removal.'
            )

        log.info(f'{self}: closing connection at {conn_req.location}')
        with self._comms_details_lock:
            # Make sure to not just remove the connection, but also to close it.
            self._comms_details.pop(conn_req.location).conn.close()

    def _process_medium(self):
        """Process the transceivers for transmission events and medium sampling, storing
        samples of the transmissions in the applicable buffers with the applicable delays.
        """
        # Get and store current time in nanoseconds.
        current_ns = monotonic_ns()

        self._maybe_advance_rx_buffers(current_ns)

        # Process receptions and transmissions
        with self._comms_details_lock:
            for loc, cmgr in self._comms_details.items():
                self._process_tx_rx_events(loc, cmgr, current_ns)

    def _maybe_advance_rx_buffers(self, current_ns: int):
        """If the current time is greater than the next slot boundary, advance the buffers
        of all :class:`Transceiver`s and update the last transition time.

        :param current_ns: The current time in nanoseconds.
        """
        if current_ns - self._last_transition_ns < self._slot_ns:
            return

        log.debug(f'{self}: crossed slot boundary, advancing rx buffers')
        self._last_transition_ns += self._slot_ns

        with self._comms_details_lock:
            for cmgr in self._comms_details.values():
                cmgr.advance_rx_buffer()

    def _process_tx_rx_events(self, loc: Location, cmgr: CommsManager, current_ns: int):
        """Process the transmission and reception events for a single
        :class:`Transceiver`.

        :param loc: The location of the :class:`Transceiver`.
        :param cmgr: The :class:`CommsManager` for the :class:`Transceiver`.
        :param current_ns: The current time in nanoseconds.
        """
        # Check if the scheduled time for the next sampling event for this transceiver has
        # passed.
        if cmgr.next_rx_ns <= current_ns:
            log.debug(f'{self}: triggering rx at location={loc}')
            try:
                cmgr.trigger_rx()
            except ConnectionError as e:
                log.error(f'{self}: error while triggering rx: {e}')

        # Check if the scheduled time for the next transmission event for this transceiver
        # has passed.
        if (tx_ns := cmgr.next_tx_ns) > current_ns:
            # Nothing to transmit yet in this time slot, so we can exit early.
            return

        # We need to pay attention to the time at which the transceiver is _scheduled_ to
        # transmit, not the current time. This value affects the slot offsets for the
        # buffers in all other transceivers.
        tx_offset_ns = tx_ns - self._last_transition_ns

        # Get the symbol that the transceiver is scheduled to transmit at this time slot
        # as well as the time it takes for the transmitter to put a symbol on the link.
        try:
            symbol, duration_ns = cmgr.trigger_tx()
        except ConnectionError as e:
            log.error(f'{self}: error while triggering tx: {e}')
            return

        log.debug(f'{self}: transceiver at location={loc} transmitting {symbol=}')

        # Add the symbol to the buffer of all other transceivers.
        # NOTE: we already have the lock, so no need to obtain it again.
        for dest_loc, dest_cmgr in self._comms_details.items():
            if dest_cmgr is cmgr:
                continue

            # Add the symbol to the buffer associated with the destination, accounting for
            #   1. the delay from the start of the time slot to the time at which the
            #      transceiver is scheduled to transmit
            #   2. the travel time between the two transceivers
            #   3. the rate at which the source transceiver is transmitting
            # With 1 and 2 accounting for when the symbol is received by the destination
            # transceiver, and 3 accounting for how long the symbol is on the link.
            log.debug(f'{self}: adding {symbol=} to buffer for location={dest_loc}')
            dest_cmgr.modify_buffer(
                symbol,
                tx_offset_ns + self._calculate_travel_time_ns(loc, dest_loc),
                duration_ns,
            )

    @classmethod
    @lru_cache
    def _calculate_travel_time_ns(self, src_loc: Location, dest_loc: Location) -> float:
        """Calculate the time (in nanoseconds) it takes for a signal to travel from one
        location to another in this medium."""
        return euclidean_distance(src_loc, dest_loc) / self._medium_velocity_ns

    # endregion

    # region Public methods

    def connect(self, conn: Connection, location: Location = None, **kwargs):
        """Communicate with the connection worker thread to establish a connection with a
        :class:`Transceiver`.

        :param conn: our :class:`~multiprocessing.Connection` to the
            :class:`~multiprocessing.Pipe` for communicating with the :class:`Transceiver`
        :param location: the desired location of the :class:`Transceiver` in the medium.
            If not provided, the :class:`Transceiver` will be placed at an location
            determined by the medium.
        """
        self._connection_queue.put(ConnRequest(conn, location, True, kwargs))

    def disconnect(self, location: Location, **kwargs):
        """Communicate with the connection worker thread to disconnect a
        :class:`Transceiver`.

        NOTE:
        At some point we may want to add a check to make sure that the transceiver
        disconnecting is the one that connected.

        :param location: the location of the :class:`Transceiver` in the medium.
        """
        self._connection_queue.put(ConnRequest(None, location, False, kwargs))

    def stop(self):
        """Let the worker thread break out of its processing loop in addition to the
        usual work done by :meth:`~multiprocessing.Process.terminate`.
        TODO:
        Think of some way to also disconnect all of the connected transceivers. Initially
        I was thinking that we could replace the location with the transceiver instances
        themselves, but I'll need to check that this is even possible for IPC."""
        self._stop_event.set()

        try:
            self._connection_queue.put(ManagementMessages.CLOSE)
        except ValueError:
            # This can happen if the queue has been closed
            pass

        # Because we can't start a new process after closing the current one, we
        # need to remove this instance from the dict of instances.
        # NOTE:
        # It may be that some other process has already performed this step, so we should
        # be robust to that.
        self._instances.pop((self.__class__, self.name), None)

        if self.is_alive():
            super().terminate()
            self.join()

    # endregion


class Transceiver(Process, ABC):
    """A base class for all transceivers. A transceiver is a device that can send and
    receive data over a physical medium. In PyNet, a :class:`Transceiver` instance is
    responsible for connecting to a :class:`Medium` instance."""

    # Keep track of all instances so we can shut down their respective processes at exit
    _instances = WeakValueDictionary()

    # The medium types supported by a transceiver subclass.
    _supported_media: Sequence[Medium]

    # The buffers to be used for inter-layer communication.
    _tx_buffer: Array
    _rx_buffer: Array

    # region Dunders

    def __init_subclass__(
        cls, supported_media: Sequence[Medium], buffer_bytes: int
    ) -> None:
        """Make sure that the subclass has the correct attributes and that the
        `supported_media` attribute is a sequence of :class:`Medium` subclasses.

        :param supported_media: The medium types supported by the transceiver subclass.
        :param buffer_size: The size of the buffer to be used for inter-layer
            communication.
        """
        super().__init_subclass__()

        bad_media = [m for m in supported_media if not issubclass(m, Medium)]
        if bad_media:
            raise TypeError(f'{bad_media} are not subclasses of {Medium}')

        # Make sure all of the media have the same dimesnionality. It doesn't make sense
        # to have a transceiver that can connect to one type of medium that only exists
        # in 1D (e.g., coax) and another that exists in 3D (air).
        if len(set(m._dimensionality for m in supported_media)) != 1:
            raise ValueError('`supported_media` must have the same number of dimensions.')

        if buffer_bytes <= 0:
            raise ValueError('`buffer_bytes` must be a positive integer.')

        cls._supported_media = supported_media

        # In the absence of a bit array type, we'll use a byte array and do the
        # necessary bit-level operations ourselves. It is the responsibility of the
        # next layer up to write bytes to the buffer and read bytes from the buffer.
        # NOTE:
        # Because data can be sent at both positie and negative energy levels, we need
        # to use signed bytes.
        cls._tx_buffer = Array('b', [0] * buffer_bytes)
        cls._rx_buffer = Array('b', [0] * buffer_bytes)

    def __new__(cls, name: str, *args, **kwargs):
        try:
            return cls._instances[cls, name]
        except KeyError:
            pass
        obj = super().__new__(cls)
        cls._instances[cls, name] = obj
        return obj

    def __init__(self, name: str, base_baud: int, auto_start: bool = True):
        """The constructor for the :class:`Transceiver` class.

        :param name: The name of the :class:`Transceiver` instance.
        :param base_baud: The base baud rate of the :class:`Transceiver`. This is the rate
            (in symbols per second) at which the :class:`Transceiver` will transmit and
            receive. In the case of receptions, however, the actual rate may fluctuate
            depending on any algorithm used to detect the rate of the incoming signal.
        :param auto_start: Whether or not to start the :class:`Transceiver` process
            automatically. If this is set to `False`, the user will need to call the
            :meth:`~multiprocessing.Process.start` method manually.
        """
        super().__init__()

        self.name: str = name
        self._medium: Medium = None

        # The base time delta between symbols nanoseconds. We _always_ transmit in
        # in intervals of this value we _start_ to receive symbols at intervals of this
        # value. The receiving delta could float up or down depending on the algorithm
        # used to detect the rate/phase of the incoming signal.
        self._base_delta_ns: int = NANOSECONDS_PER_SECOND // base_baud

        # The type of the shared memory used for the location depends on the number of
        # axes that the location is defined over.
        if (axes_count := self._supported_media[0]._dimensionality) == 1:
            self._location: Value = Value('f', 0.0)
        else:
            self._location: Array = Array('f', [0.0] * axes_count)

        # Store the name as a value so that it can be accessed within the process.
        self._medium_name: Array = Array('c', DEFAULT_NAME)

        # The connection over which callers can make alterations to the connectivity of
        # the transceiver. The client is used to forward connection requests from the user
        # into the Tranceiver process and the listener is monitored within the process.
        conn_listener, conn_client = Pipe(duplex=False)
        self._connections_client: Connection = conn_client
        self._connections_listener: Connection = conn_listener

        # An event to let both the main Process and any worker threads know that it's
        # time to shut down.
        self._stop_event: Event = Event()

        # The pipe over which communications are sent to the next layer up. The client is
        # used by a thread monitoring the medium to pass data up to the next layer in the
        # OSI model (e.g., MAC in data link). The listener is used by the user to receive
        # data from the medium via the transceiver (which is why it is public).
        osi_listener, osi_client = Pipe(duplex=False)
        self.osi_listener: Connection = osi_listener
        self._osi_client: Connection = osi_client

        # By default, we do the caller the service of starting up the Process rather than
        # making them remember to do it for each and every Transceiver instance.
        if auto_start:
            self.start()

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

    def __repr__(self) -> str:
        details_str = f'name={self.name!r}'

        if (name := self._medium_name.value) == DEFAULT_NAME:
            details_str += ', location=disconnected'
        else:
            details_str += f', location={name.decode()}@{self.location}'

        details_str += f', pid={self.pid}'
        if self.pid and not self.is_alive():
            details_str += '(exited)'

        return f'<{self.__class__.__name__}: {details_str}>'

    # endregion

    # region Properties

    @property
    def medium(self) -> Medium:
        return self._medium

    @medium.setter
    def medium(self, medium: Medium) -> None:
        self.connect(medium)

    @property
    def location(self) -> Location:
        try:
            # Is the location a single value?
            return self._location.value
        except AttributeError:
            # If not, it must be an array.
            return tuple(self._location)

    @location.setter
    def location(self, location: Location) -> None:
        try:
            # Is the location a single value?
            self._location.value = location
        except AttributeError:
            # If not, it must be an array.
            self._location[:] = location

    # endregion

    # region Abstract methods

    @abstractmethod
    def _process_rx_value(self, symbol: float | int):
        """Process a symbol received from the medium, e.g., buffer, decode. It is up to
        the subclasses to determing how what to do with the result, but it will eventually
        be stored in the :attr:`_rx_buffer` and made available to the next layer up.
        """
        raise NotImplementedError

    @abstractmethod
    def _next_tx_symbol(self) -> float | int:
        """Provide us with the next symbol to be put on the medium, e.g., encoding. It is
        up to the subclasses to determine how to do this, but the result will be sent to
        the medium via the :attr:`_tx_queue`. For example, a :class:`Transceiver` which
        uses Manchester encoding might take the next bit from the :attr:`_tx_buffer` and
        convert it into the two symbols needed to represent it. It would then buffer the
        second symbol and return the first. On the next call, it would return the second
        symbol and buffer the second bit.
        """
        raise NotImplementedError

    # endregion

    # region Optional overrides

    def _connect(self, medium: Medium, **kwargs):
        """Class-specific connection logic"""
        pass

    def _disconnect(self):
        """Class-specific disconnection logic"""
        pass

    # endregion

    # region Multiprocessing- / IPC-related methods

    def run(self):
        log.info(f'Starting transceiver process ({self.pid})')

        thread_stop_event = Event()
        current_conn: Connection = None

        while not self._stop_event.is_set():
            # Wait for a connection update.
            new_conn = self._connections_listener.recv()
            if new_conn is ManagementMessages.CLOSE:
                break

            if new_conn is not None:
                # Start up a new thread which will use this connection to receive data,
                # process it and relay the result to the next layer up.
                thread_stop_event.clear()
                thread = Thread(
                    target=self._monitor_medium,
                    args=(new_conn,),
                )
                thread.start()

                # Store the new connection such that we may send a close message on
                # disconnection, thereby letting the thread unblock.
                current_conn = new_conn
            else:
                # Stop the thread which is currently receiving bits.
                thread_stop_event.set()
                current_conn.send(ManagementMessages.CLOSE)
                thread.join()

                current_conn.conn.close()
                current_conn = None
                self._medium = None

        log.debug(f'{self}: shutting down')

    def _monitor_medium(self, conn: Connection):
        """Watch the medium for incoming data. This method will be run in a Thread whose
        lifetime is only as long as the lifetime of the connection to the medium. That is,
        the thread will be started when the connection is first made and will be stopped
        when the connection is terminated."""
        while not self._stop_event.is_set():
            try:
                data = conn.recv()
            except Exception as e:
                log.error(f'{self}: Error receiving data from medium: {e}')
                continue

            if data is ManagementMessages.CLOSE:
                break

            # At this point we're expecting something of the form (tx/rx flag, value).
            try:
                comms_type, value = data
            except TypeError:
                log.error(
                    (
                        f'{self}: Received malformed communications event data from '
                        f'medium: {data=}'
                    )
                )
                continue

            if comms_type == CommsType.RECEIVE:
                self._process_reception_event(conn, value)
            elif comms_type == CommsType.TRANSMIT:
                self._process_transmission_event(conn)
            else:
                log.error(
                    (
                        f'{self}: Received unknown communications event type from '
                        f'medium: {comms_type=}'
                    )
                )

        # We've exited the loop. This means our work is done, and we should close the
        # connection on our end.
        conn.close()

    def _process_reception_event(self, conn: Connection, value: int | float):
        """
        It's the responsibility of the subclass to determine what to do with the value
        (e.g., how to manipulate/decode it, where to put it). Addtionally, the subclass
        must give us the amount of nanoseconds to wait before sampling the :class:`Medium`
        again.
        """
        try:
            next_rx_delta = self._process_rx_value(value)
        except Exception as e:
            log.error(f'{self}: Error processing {value=}: {e}')
            # Regardless of the fact that there was an error, we should still be
            # configuring the next reception time, in this case defaulting to the
            # base delta.
            next_rx_delta = self._base_delta_ns

        # Reply with the next reception time.
        conn.send(next_rx_delta)

    def _process_transmission_event(self, conn: Connection):
        """
        The :class:`Medium` is telling us that we must now put a symbol on the
        :class:`Medium`. It's the responsibility of the subclass to determine what to do
        with the symbol (e.g., how to manipulate/encode it, where to get it from).
        """
        try:
            symbol = self._next_tx_symbol()
        except Exception as e:
            log.error(f'{self}: Error getting next symbol to transmit: {e}')
            # The medium is waiting on us to send it a symbol, so we must send
            # something. We'll send a 0 to indicate that we have nothing to send.
            symbol = 0

        # Send the symbol to the medium as well as the length of the symbol in
        # nanoseconds and the amount of time to wait before sending the next
        # symbol.
        # TODO:
        # For now we'll assume that the inter-symbol time and the length of the
        # symbol are constant. A future improvement is to allow for phase shifts
        # and clock drift(?).
        try:
            conn.send(symbol, self._base_delta_ns, self._base_delta_ns)
        except Exception as e:
            log.error(f'{self}: Error sending {symbol=} to medium: {e}')

    # endregion

    # region Public methods

    def connect(self, new_medium: Medium, **kwargs) -> None:
        """Connect this :class:`Transceiver` to a :class:`Medium`, alerting the main
        process that a connection has been made and starting a thread to monitor the
        medium for incoming data.

        :param new_medium: The medium to connect to.
        """
        if not self.is_alive():
            raise ProcessNotRunningError(
                'Transceiver must be running before connecting to a Medium'
            )

        if not new_medium:
            # :meth:`Transceiver.connect` could have been called via the setter method
            # with a `None` value. We want to let this happen without raising an error,
            # but we should log the re-routing for debugging purposes.
            log.debug('Connecting to medium=None. Assuming disconnect')
            self.disconnect()
            return

        # Nothing to do if the incoming is the same medium being connected, otherwise
        # raise an exception.
        if current_medium := self._medium:
            if new_medium is current_medium:
                log.debug(f'{self!r} already connected to {new_medium!r}')
                return
            else:
                raise RuntimeError(
                    f'{self!r} already connected to {current_medium!r}. Disconnect first'
                )

        # Make sure to check the validity of the incoming medium.
        if (medium_cls := new_medium.__class__) not in self._supported_media:
            raise ValueError(f'Medium {medium_cls.__name__} not supported by {self!r}')

        if not new_medium.is_alive():
            # TODO:
            # Get rid of this check if we refactor Medium to subclass Queue.
            raise ProcessNotRunningError(
                'Transceivers may only connect to running Mediums'
            )

        log.debug(f'Connecting {self!r} to {new_medium!r}')

        # Let the subclass runs its connection logic (maybe it wants to object to the
        # connection).
        self._connect(new_medium, **kwargs)

        # The subclass is okay with the connection, so create pipes for transmission and
        # receptions with respect to link (i.e., medium <-> transceiver).
        xcvr_conn, medium_conn = Pipe(duplex=True)

        # Give one end of the pipe to the medium, the reception of which will trigger
        # its connection logic.
        log.info(f'Sending connection request to {new_medium!r}')
        try:
            new_medium.connect(medium_conn, **kwargs)
        except Exception as e:
            raise ConnectionError(
                f'Error sending connection request to {new_medium!r}: {e}'
            )

        # Wait for the medium to acknowledge the connection with a message containing our
        # location.
        try:
            response = xcvr_conn.recv()
        except Exception as e:
            raise ConnectionError(
                f'Error receiving connection response from {new_medium!r}: {e}'
            )

        try:
            result, location = response
        except ValueError:
            raise ConnectionError(
                f'Unexpected response contents from {new_medium!r}: {response!r}'
            )

        if result is not Responses.OK:
            raise ConnectionError(
                f'{new_medium!r} rejected connection request: {result=}, details='
                f'{location!r}'
            )

        # The medium has acknowledged the connection, meaning that we are now connected
        # and can set our location and set up a thread to monitor the medium for incoming
        # data.
        self.location = location

        # Configure ourselves with the other ends of the pipe.
        # NOTE:
        # Because this method is being accessed outside of the process, we need to use
        # our own connection pipe to communicate the change to the process.
        self._connections_client.send(xcvr_conn)

        # Keep track of the medium so we can disconnect from it later or prevent
        # the changes seen at the top of this method.
        self._medium = new_medium
        # Save the name separately in the array so that the process can access it as well.
        self._medium_name.value = new_medium.name.encode()

    def disconnect(self) -> None:
        """Disconnect this :class:`Transceiver` from its current :class:`Medium`, alerting
        the medium to the fact that the connection is being terminated.
        """
        if not (medium := self._medium):
            # Already disconnected
            return

        # No need to specify the medium here since :meth:`__repr__` will return the
        # medium detatils.
        log.debug(f'Disconnecting {self!r} from {medium!r}')

        # Let the subclass run its disconnection logic.
        self._disconnect()

        # Alert the medium to the fact that we are disconnecting.
        medium.disconnect(self.location)

        # Finally, alert the process to the fact that we are disconnecting.
        # NOTE:
        # Because this method is being accessed outside of the process, we need to use
        # our own connection to communicate the change to the process.
        self._connections_client.send(None)

        self._medium = None
        self._medium_name.value = DEFAULT_NAME

    def stop(self):
        """Gracefully stop the transceiver process"""
        # Tell the process to get rid of any existing connection to a medium.
        self.disconnect()

        # # Let the process know that it should stop.
        self._stop_event.set()
        try:
            self._connections_client.send(ManagementMessages.CLOSE)
        except OSError:
            # The pipe is already closed.
            pass

        # Because we can't start a new process after closing the current one, we
        # need to remove this instance from the dict of instances.
        # NOTE:
        # It may be that some other process has already performed this step, so we should
        # be robust to that.
        self._instances.pop((self.__class__, self.name), None)

        if self.is_alive():
            super().terminate()
            self.join()

    # endregion


# Be sure to clean up the :class:`Transceiver` and :class:`Medium` processes at exit.
@atexit.register
def cleanup_processes():
    for cls in (Transceiver, Medium):
        for instance in cls._instances.values():
            try:
                instance.terminate()
            except AttributeError:
                continue
            instance.join()
