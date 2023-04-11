"""
Base classes for physical media and transceivers, with these classes handling the
low-level details of connecting and disconnecting transceivers from media. Subclasses
of these base classes define the specifics of how transceivers connect to media and
how data is transmitted over the media.

:author: Alf O'Kenney

TODO:
- active bit in transceiver
- add in-process init to allow users to initialize certain values that only really need to
  be used within the process
    - if they are created in __init__ then they must be Value or Array objects, which is
      inconvenient
    - additionally, users would need to remember to define these variabes prior to calling
        super().__init__() in their own __init__ methods such that they would be available
        after the fork
- allow users to specify the clock frequency of the transceivers
- allow users to specify a PLL for transceivers
- implement subclass abstract base classes for electrical and optical media
- maybe further divide them into 1D and 3D media
- configure logging (esp. for processes)
- allow users to define the environment (e.g., temperature, humidity, walls, buildings,
    etc.) in which the transceivers and media are located
        - this will have on impact on what, if anything is received by devices
- make the medium robust to delays so that we can decrease the time dilation factor
"""

from __future__ import annotations

import atexit
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import cached_property, lru_cache
from multiprocessing import Array, Event, Lock, Pipe, Process, Queue
from multiprocessing.connection import Connection
from threading import Thread
from time import monotonic_ns
from typing import Any, Callable, Sequence
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
from .exceptions import ConnectionError, ProcessNotRunningError, TransmissionComplete
from ..space import Location, euclidean_distance


__all__ = ['Medium', 'Transceiver']

log = logging.getLogger(__name__)

Number = int | float

ConnRequest = namedtuple('ConnRequest', ['conn', 'location', 'create', 'kwargs'])


class Transmission:
    """A class for describing a transmission with respect to two :class:`Transceiver`
    objects."""

    def __init__(
        self,
        symbol: Number | Callable,
        start_ns: float,
        duration_ns: float,
        attenuation: float,
    ):
        """
        :param symbol: Either a static amplitude or a function that takes the time since
            the start of the transmission as its input and returns the amplitude of the
            signal at that time.
        :param start_ns: The global time (in nanoseconds) at which the transmission
            begins.
        :param duration_ns: The duration of the transmission in nanoseconds.
        :param attenuation: The attenuation of the transmission at the reference point
            associated with the reception of the transmission.
        """
        self._symbol = symbol
        self._start_ns = start_ns
        self._stop_ns = start_ns + duration_ns * TIME_DILATION_FACTOR
        self._attenuation = attenuation

    @cached_property
    def _symbol_fn(self) -> Callable[[Number], float]:
        """A function that takes a single argument, the time since the start of the
        transmission, and returns the amplitude of the symbol at that time.
        """
        if isinstance(self._symbol, (int, float)):
            return lambda _: self._symbol
        return self._symbol

    def get_amplitude(self, time_ns: float) -> float:
        """Read the amplitude of the symbol at a given time.

        :param time_ns: The time in nanoseconds at which to read the amplitude.

        :returns: The amplitude of the signal at the given time.
        """
        if time_ns < self._start_ns:
            return 0
        if time_ns >= self._stop_ns:
            raise TransmissionComplete

        # We need to de-dilate the time to get the correct amplitude because the main
        # medium process is sampling at a slower rate than the transceivers would be in
        # real life. That is, the difference between the start time and the current time
        # is TIME_DILATION_FACTOR times larger than it would be in real life.
        actual_ns = (time_ns - self._start_ns) / TIME_DILATION_FACTOR
        return self._symbol_fn(actual_ns) * self._attenuation


class CommsManager:
    """A container for managing the relationship between a :class:`Transceiver` and a
    :class:`Medium` which the latter uses to trigger all communications (i.e., both TX and
    RX) with the former.
    """

    def __init__(self, conn: Connection):
        """
        :param conn: The :class:`Medium`'s :class:`.Connection` of the :class:`.Pipe`
            between itself and a :class:`Transceiver`.
        """
        self._conn: Connection = conn
        self._transmissions: set[Transmission] = set()

        # Start the receiver off by waiting for an active medium.
        self.next_rx_ns = -1

    def _send(self, data: Any) -> None:
        """Send data to the :class:`Transceiver`.

        :param data: The data to send to the :class:`Transceiver`.
        """
        try:
            self._conn.send(data)
        except (BrokenPipeError, ConnectionResetError):
            # Alert the higher layers to the fact that the connection has very likely
            # closed.
            raise
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
        except (BrokenPipeError, ConnectionResetError):
            # Alert the higher layers to the fact that the connection has very likely
            # closed.
            raise
        except Exception as e:
            raise ConnectionError(
                f'Could not receive data from the Transceiver via {self}: {e}'
            )

    def trigger_tx(self) -> tuple[Callable | None, float]:
        """Ping the :class:`Transceiver` to transmit a symbol and wait for a response
        containing the function describing the symbol to be transmitted and the duration
        of the symbol in nanoseconds.

        :returns: A tuple containing:
            - a function describing the amplitude of a symbol over time
            - the duration in nanoseconds for the symbol to be put on the link.
        """
        self._send((CommsType.TRANSMIT, None))

        return self._recv()

    def maybe_trigger_rx(self, current_time_ns: float) -> None:
        """Send the summation of the various symbol amplitudes at this time to the
        :class:`Transceiver`.

        :param current_time_ns: The time in nanoseconds from the pespective of the
            :class:`Medium`.

        TODO:
        Gracefully recover from an exception by using the previous delta time.
        """
        amplitude = self._get_amplitude(current_time_ns)

        if self.next_rx_ns == -1 and amplitude == 0:
            # The medium is idle and the receiver is not in sampling mode, so there is no
            # need to trigger a reception.
            return

        self._send((CommsType.RECEIVE, amplitude))

        # The transceiver will respond with the delta time to the next rx event.
        rx_delta_ns = self._recv()

        if rx_delta_ns == -1:
            # The transceiver wishes to only be contacted when the medium is no longer
            # idle.
            # NOTE:
            # We set to -1 so that the medium will always attempt to trigger a reception.
            self.next_rx_ns = -1
        else:
            # The transceiver wants to keep sampling, so update the next rx time.
            if self.next_rx_ns == -1:
                # The medium was previously idle, so there was no previous sample time and
                # we need to instead offset from the current time.
                this_rx_ns = current_time_ns
            else:
                # We're continuing from a previous sample time.
                this_rx_ns = self.next_rx_ns

            # NOTE:
            # We need to scale up the time to allow us to operate at a slower rate than
            # the transceivers would be in real life. Don't worry, the Transmission object
            # will scale back down again when it reads from the symbol function.
            self.next_rx_ns = this_rx_ns + rx_delta_ns * TIME_DILATION_FACTOR

    def _get_amplitude(self, current_time_ns: float):
        # Collect the summation of the current amplitudes of all of the transmissions.
        amplitude = 0.0
        sample_time = self.next_rx_ns if self.next_rx_ns != -1 else current_time_ns

        # Work with a shallow copy so that we can remove transmissions from the original
        # set as we go.
        for tx in self._transmissions.copy():
            try:
                amplitude += tx.get_amplitude(sample_time)
            except TransmissionComplete:
                # The transmission has finished, so we must remove it from the set.
                self._transmissions.remove(tx)
            except Exception as e:
                # Leave the transmission in the set and log the error. The transmission
                # class itself will tell us when it's time to remove it.
                log.error(
                    f'{self}: an exception occurred while processing a transmission: {e}'
                )

        return amplitude

    def add_transmission(
        self,
        symbol: Callable | Number,
        start_ns: float,
        duration_ns: float,
        attenuation: float = 1.0,
    ):
        """Used by the :class:`Medium` to update the set of transmissions to be received
        by the :class:`Transceiver`.

        :param symbol: Either a static amplitude or a function that takes the time since
            the start of the transmission as its input and returns the amplitude of the
            signal at that time.
        :param start_ns: The global time in nanoseconds at which the transmission begins.
        :param duration_ns: The duration of the transmission in nanoseconds.
        :param attenuation: The attenuation of the transmission once it reaches the
            :class:`Transceiver`.
        """
        self._transmissions.add(Transmission(symbol, start_ns, duration_ns, attenuation))


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
        :param velocity_factor: The ratio of the speed of a wave propagating along the
            medium to the speed of light. Must be in range (0, 1].
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

    def __init__(self, name: str, auto_start: bool = True):
        """Initialize a :class:`Medium` instance.

        :param name: The name of the medium. This is used to identify the medium when
            connecting :class:`Transceiver` instances to it.
        :param auto_start: Whether or not to automatically start the process when the
            instance is created. Defaults to ``True``.
        """
        super().__init__()

        self.name = name

        self._connection_thread: Thread = None

        # The dict of details about connected transceiver, and the lock used to ensure
        # thread safety.
        self._comms_details: dict[Location, CommsManager] = {}
        self._comms_details_lock: Lock = Lock()

        # The Queue monitored by the worker thread for updates to state of connected
        # Transceivers.
        self._connection_queue: Queue = Queue()

        # This event is used to stop both ourselves (the forked Process) and Thread
        # workers.
        self._stop_event: Event = Event()

        # At any given moment, one or more Transceivers may be transmitting a signal. We
        # need to keep track of the completion times of these signals in order to:
        #   1. avoid unnecessary polling of transceivers that are already transmitting
        #   2. replace a completed symbol transmission with a new one (if available)
        self._transmission_completion_times: dict[CommsManager, int] = {}

        self._init_shared_objects()

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

    def _init_shared_objects(self):
        """Run any initialization logic required before the process starts. Users can
        always define an `__init__` method that ensures shared object initialization
        before calling `super().__init__`, but this method makes that process
        idiot-proof."""
        pass

    def _disconnect(self, *args, **kwargs):
        """Run class-specific disconnection logic"""
        pass

    # endregion

    # region Multi-processing- / IPC-related methods

    def run(self):
        log.info(f'Starting medium process ({self.pid})')

        # Start the worker thread to monitor connection changes.
        self._connection_thread = Thread(target=self._monitor_connections)
        self._connection_thread.start()

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

        new_cmgr = CommsManager(conn)

        with self._comms_details_lock:
            self._comms_details[location] = new_cmgr

        # Send the location back to the transceiver to confirm that everything is okay.
        try:
            conn.send((Responses.OK, location))
        except Exception as e:
            log.error(f'{self}: connection info reply to transceiver failed: {e}')

            # We need to remove the connection we just added.
            # NOTE:
            # The order might be odd, but it prevents race conditions where transceivers
            # begin to transmit immediately after the connection is established and other
            # transceivers are still being added in a loop
            with self._comms_details_lock:
                del self._comms_details[location]
            return

        log.info(f'{self}: connection established at {location=}')

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

        log.info(f'{self}: closing connection at location={conn_req.location}')
        with self._comms_details_lock:
            # Make sure to not just remove the connection, but also to close it.
            self._comms_details.pop(conn_req.location)._conn.close()

    def _process_medium(self):
        """Process the transceivers for transmission events and medium sampling, storing
        transmission signal functions in the applicable receivers with the applicable
        delays.
        """
        # Get and store current time in nanoseconds.
        current_ns = monotonic_ns()

        # Process receptions and transmissions
        with self._comms_details_lock:
            # NOTE:
            # We MUST check TX events of _all_ transceivers before checking _any_ RX
            # events, as we may need to clean up our view of ongoing transmissions and add
            # new ones to the RX queues. For example, the 5th transceiver in the dict may
            # be finished transmitting its first symbol and needs to start transmitting
            # the next, but if we have a single loop going over the TX and RX events of
            # each transceiver in the order of the dict, then we will necessarily process
            # the receptions of the 0th transceiver before we get the new transmission
            # from the 5th transceiver.
            for loc, cmgr in self._comms_details.items():
                self._process_tx_events(loc, cmgr, current_ns)
            for loc, cmgr in self._comms_details.items():
                self._process_rx_events(loc, cmgr, current_ns)

    def _process_tx_events(self, loc: Location, cmgr: CommsManager, current_ns: int):
        """Process the transmission events for a single :class:`Transceiver`.

        :param loc: The location of the :class:`Transceiver`.
        :param cmgr: The :class:`CommsManager` for the :class:`Transceiver`.
        :param current_ns: The current time in nanoseconds.
        """
        # Check if the transceiver had been transmitting and if the transmission has
        # finished.
        try:
            stop_time_ns = self._transmission_completion_times[cmgr]
        except KeyError:
            # The transceiver is not transmitting, so we can assume that if it has
            # anything to transmit, it just started.
            start_ns = current_ns
        else:
            # The transceiver had been transmitting. Now we need to check if it has
            # completed the transmission.
            if stop_time_ns <= current_ns:
                del self._transmission_completion_times[cmgr]

                # If there is another symbol to transmit, we need to schedule the
                # transmission of that symbol immediately after the end of the current
                # transmission.
                start_ns = stop_time_ns
            else:
                # The transceiver is still transmitting.
                return

        # Check if the transceiver has any symbols to transmit.
        try:
            symbol_info = cmgr.trigger_tx()
        except (BrokenPipeError, ConnectionResetError) as e:
            # No need for a higher level, since this is a normal occurrence triggered
            # by a shutdown race condition.
            log.debug(f'{self}: connection to transceiver at {loc} closed: {e}')
            return
        except ConnectionError as e:
            log.error(f'{self}: error while triggering tx: {e}')
            return

        if symbol_info is None or symbol_info is ManagementMessages.CLOSE:
            # The transceiver has nothing to transmit.
            return

        # Oh, it's transmitting, alright.
        symbol, duration_ns = symbol_info

        log.debug(
            f'{self}: transceiver at location={loc} beginning transmission of new symbol'
        )

        # Keep track of the transmission so that we can check if it has finished.
        self._transmission_completion_times[cmgr] = (
            start_ns + duration_ns * TIME_DILATION_FACTOR
        )

        # Add the symbol to the set of transmission functions of each of the other
        # transceivers.
        for dest_loc, dest_cmgr in self._comms_details.items():
            if dest_cmgr is cmgr:
                continue

            # TODO:
            # Allow for a calculation of signal attenuation based on distance.
            log.debug(f'{self}: adding transmission to set for location={dest_loc}')
            dest_cmgr.add_transmission(
                symbol,
                start_ns + self._calculate_travel_time_ns(loc, dest_loc),
                duration_ns,
                1,  # No attenuation for now.
            )

    def _process_rx_events(self, loc: Location, cmgr: CommsManager, current_ns: int):
        # Check if the scheduled time for the next sampling event for this transceiver has
        # passed.
        if cmgr.next_rx_ns > current_ns:
            return

        log.debug(f'{self}: maybe triggering rx at location={loc}')
        try:
            # Let the CommsManager know the current time in case the Transceiver was
            # not sampling and was instead waiting to sense a transmission on the
            # medium.
            cmgr.maybe_trigger_rx(current_ns)
        except (BrokenPipeError, ConnectionResetError) as e:
            # No need for a higher level, since this is a normal occurrence triggered
            # by a shutdown race condition.
            log.debug(f'{self}: connection to transceiver at {loc} closed: {e}')
        except ConnectionError as e:
            log.error(f'{self}: error while triggering rx: {e}')

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

    # The size of the buffers to be used for inter-layer communication.
    _buffer_bytes: int

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
            raise TypeError(f'{bad_media} are not subclasses of Medium')

        # Make sure all of the media have the same dimesnionality. It doesn't make sense
        # to have a transceiver that can connect to one type of medium that only exists
        # in 1D (e.g., coax) and another that exists in 3D (air).
        if len(set(m._dimensionality for m in supported_media)) != 1:
            raise ValueError('`supported_media` must have the same number of dimensions.')

        if buffer_bytes <= 0:
            raise ValueError('`buffer_bytes` must be a positive integer.')

        cls._supported_media = supported_media
        cls._buffer_bytes = buffer_bytes

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

        # The base time delta between symbols nanoseconds. We start to receive symbols at
        # intervals of this value. The receiving delta could float up or down depending on
        # the algorithm used to detect the rate/phase of the incoming signal.
        self._base_delta_ns: int = NANOSECONDS_PER_SECOND // base_baud

        self._location: Array = Array(
            'f', [0.0] * self._supported_media[0]._dimensionality
        )

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

        # The buffers to be used for inter-layer communication.
        # NOTE:
        # - These are not used within this parent class. The intention is for them to be
        #   used as a workspace for child classes for communicating between layers. For
        #   example, buffering two symbols of a Manchester encoded bit from the medium and
        #   placing this decoded bit in the rx buffer before signaling to the MAC layer
        #   that the full frame has been received.
        # - In the absence of a bit array type, we'll use a byte array and do the
        #   necessary bit-level operations ourselves. It is the responsibility of the next
        #   layer up to write bytes to the buffer and read bytes from the buffer.
        # - Because data can be sent at both positive and negative energy levels, we need
        #   to use signed bytes.
        self._tx_buffer: Array = Array('b', [0] * self._buffer_bytes)
        self._rx_buffer: Array = Array('b', [0] * self._buffer_bytes)

        self._init_shared_objects()

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
        loc = self._location[:]

        if len(loc) == 1:
            return loc[0]
        return tuple(loc)

    @location.setter
    def location(self, location: Location) -> None:
        # We can't rely on an AttributeError being raised if the location is a single
        # value, so we'll go with an LBYL approach.
        current_location = self._location[:]
        expected_len = len(current_location)

        if isinstance(location, (float, int)):
            if expected_len == 3:
                raise ValueError('Expected a sequence of length 3, got a scalar instead')
            self._location[0] = location
        elif isinstance(location, (list, tuple)):
            if (bad_len := len(location)) != expected_len:
                raise ValueError(
                    f'Expected a sequence of length {expected_len}, got sequence of length {bad_len} instead'
                )
            self._location[:] = location
        else:
            raise TypeError(
                f'Expected a sequence or scalar for location, got {type(location).__name__} instead'
            )

    # endregion

    # region Abstract methods

    @abstractmethod
    def _process_rx_amplitude(self, amplitude: Number):
        """Process the amplitude value received from the medium. It is up to the
        subclasses to determine what to do with the result (e.g., buffer, decode), and it
        will eventually be stored in the :attr:`_rx_buffer` and made available to the next
        layer up.
        """
        raise NotImplementedError

    @abstractmethod
    def _next_tx_symbol(self) -> Number | None:
        """Provide us with the next symbol to be put on the medium. It is up to the
        subclasses to determine how to do this (e.g., encoding), and the result will be
        sent to the medium via the :attr:`_tx_queue`. For example, a :class:`Transceiver`
        which uses Manchester encoding might take the next bit from the :attr:`_tx_buffer`
        and convert it into the two symbols needed to represent it. It would then buffer
        the second symbol and return the first. On the next call, it would return the
        second symbol and buffer the next bit in the data.

        If the :attr:`_tx_buffer` is empty, this method should return `None` to indicate
        that there is no more data to send, since a 0 could be a valid symbol.
        """
        raise NotImplementedError

    # endregion

    # region Optional overrides

    def _init_shared_objects(self):
        """Run any initialization logic required before the process starts. Users can
        always define an `__init__` method that ensures shared object initialization
        before calling `super().__init__`, but this method makes that process
        idiot-proof."""
        pass

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

        current_conn: Connection = None

        while not self._stop_event.is_set():
            # Wait for a connection update.
            new_conn = self._connections_listener.recv()
            if new_conn is ManagementMessages.CLOSE:
                break

            if new_conn is not None:
                # Start up a new thread which will use this connection to receive data,
                # process it and relay the result to the next layer up.
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
                current_conn.send(ManagementMessages.CLOSE)
                thread.join()

                current_conn.close()
                current_conn = None
                self._medium = None

        # If we're exiting the loop, we need to make sure that the thread is joined.
        if current_conn is not None:
            thread.join()

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

            # At this point we're expecting something of the form (tx/rx flag, amplitude).
            try:
                comms_type, amplitude = data
            except TypeError:
                log.error(
                    (
                        f'{self}: Received malformed communications event data from '
                        f'medium: {data=}'
                    )
                )
                continue

            if comms_type == CommsType.RECEIVE:
                self._process_reception_event(conn, amplitude)
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

    def _process_reception_event(self, conn: Connection, amplitude: Number):
        """
        It's the responsibility of the subclass to determine what to do with the observed
        amplitude (e.g., how to manipulate/decode it, where to put it). Addtionally, the
        subclass must give us the amount of nanoseconds to wait before sampling the
        :class:`Medium` again, with a value of -1 indicating that we should only be
        informed of the next reception event (simulating the monitoring of the "active"
        bit).

        :param conn: The connection to the medium.
        :param amplitude: The amplitude value received from the medium.
        """
        try:
            next_rx_delta = self._process_rx_amplitude(amplitude)
        except Exception as e:
            log.error(f'{self}: Error processing {amplitude=}: {e}')
            # Regardless of the fact that there was an error, we should still be
            # configuring the next reception time, in this case defaulting to the
            # base delta.
            next_rx_delta = self._base_delta_ns

        if next_rx_delta is None:
            # If the subclass didn't give us a next reception time, we'll just default to
            # half of the base delta, in keeping with the Nyquist-Shannon sampling
            # theorem.
            next_rx_delta = self._base_delta_ns / 2

        # Reply with the next reception time.
        conn.send(next_rx_delta)

    def _process_transmission_event(self, conn: Connection):
        """
        The :class:`Medium` is telling us that we must now put a symbol on the
        :class:`Medium`. It's the responsibility of the subclass to determine how to
        create the symbol (e.g., how to manipulate/encode it, where to get it from). All
        we do is make sure to send a representative function to the :class:`Medium`.
        """
        try:
            symbol = self._next_tx_symbol()
        except Exception as e:
            log.error(f'{self}: Error getting next symbol to transmit: {e}')

            # Tell the medium to move on without us.
            conn.send(None)
            return

        if symbol is None:
            # We don't have anything to transmit, so we'll tell the medium to move on
            # without us.
            conn.send(None)
            return

        # We could use EAFP to check if this is a function, but it's possible that an
        # AttributeError will be raised by the function itself, which we should catch
        # before passing this on the medium. So we'll use LBYL.
        if callable(symbol):
            # The output must be a float or integer.
            try:
                amplitude = symbol(0)
            except Exception as e:
                log.error(f'{self}: Error getting amplitude from symbol function: {e}')

                # Tell the medium to move on without us.
                conn.send(None)
                return

            if not isinstance(amplitude, (int, float)):
                log.error(
                    f'{self}: Symbol function returned an amplitude of type '
                    f'{amplitude.__class__.__name__} ({amplitude!r}); expected int or '
                    'float'
                )

                # Tell the medium to move on without us.
                conn.send(None)
                return
        else:
            # Okay, so it's not a function. We'll assume it's a float or integer.
            if not isinstance(symbol, (int, float)):
                log.error(
                    f'{self}: Symbol is of type {symbol.__class__.__name__} '
                    f'({symbol!r}); expected int, float or callable'
                )

                # Tell the medium to move on without us.
                conn.send(None)
                return

        # Send the symbol function to the medium as well as the length of the symbol in
        # nanoseconds
        try:
            conn.send((symbol, self._base_delta_ns))
        except Exception as e:
            log.error(f'{self}: Error sending symbol function to medium: {e}')

    # endregion

    # region Buffer interaction

    def _set_rx_buffer_data(self, data: Sequence | int, start_index: int = 0) -> None:
        """Set the contents of the receive buffer.
        NOTE:
        This only overwrites the contents of the receive buffer from the start index to
        the length of the incoming data. If the incoming data is shorter than the current
        contents of the receive buffer, the remaining contents will be left unchanged.

        :param data: The data to use to replace the current contents of the receive
            buffer.
        :param start_index: The index at which to start writing the incoming data.
        """
        end_index = len(data) + start_index
        self._rx_buffer[start_index:end_index] = data

    def _set_rx_buffer_element(self, value: int, index: int) -> None:
        """Set a single element in the receive buffer

        :param value: The value to set the element to.
        :param index: The index of the element to set.
        """
        self._rx_buffer[index] = value

    def clear_rx_buffer(self) -> None:
        """Clear the contents of the receive buffer"""
        self._rx_buffer[:] = [0] * len(self._rx_buffer)

    def get_rx_buffer_data(self, start_index: int = 0, end_index: int = None) -> list:
        """Get the contents of the receive buffer

        :param start_index: The index at which to start reading the receive buffer.
        :param end_index: The index at which to stop reading the receive buffer. If not
            provided, the remainder of the receive buffer will be returned.
        :return: The contents of the receive buffer
        """
        end_index = end_index or len(self._tx_buffer)
        return self._rx_buffer[start_index:end_index]

    def get_rx_buffer_element(self, index: int) -> int:
        """Get a single element from the receive buffer

        :param index: The index of the element to get.
        :return: The element at the given index.
        """
        return self._rx_buffer[index]

    def set_tx_buffer_data(self, data: Sequence, start_index: int = 0) -> None:
        """Set the contents of the transmit buffer
        NOTE:
        This only overwrites the contents of the transmit buffer from the start index to
        the length of the incoming data. If the incoming data is shorter than the current
        contents of the transmit buffer, the remaining contents will be left unchanged.

        :param data: The data to use to replace the current contents of the transmit
            buffer.
        :param start_index: The index at which to start writing the incoming data.
        """
        end_index = len(data) + start_index
        self._tx_buffer[start_index:end_index] = data

    def set_tx_buffer_element(self, value: int, index: int) -> None:
        """Set a single element in the transmit buffer

        :param value: The value to set the element to.
        :param index: The index of the element to set.
        """
        self._tx_buffer[index] = value

    def get_tx_buffer_data(self, start_index: int = 0, end_index: int = None) -> list:
        """Get the contents of the transmit buffer

        :param start_index: The index at which to start reading the transmit buffer.
        :param end_index: The index at which to stop reading the transmit buffer. If not
            provided, the remainder of the transmit buffer will be returned.
        :return: The contents of the transmit buffer
        """
        end_index = end_index or len(self._tx_buffer)
        return self._tx_buffer[start_index:end_index]

    def get_tx_buffer_element(self, index: int) -> int:
        """Get a single element from the transmit buffer

        :param index: The index of the element to retrieve.
        :return: The element at the given index.
        """
        return self._tx_buffer[index]

    def clear_tx_buffer(self) -> None:
        """Clear the contents of the transmit buffer"""
        self._tx_buffer[:] = [0] * len(self._tx_buffer)

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
