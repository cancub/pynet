"""
Base classes for physical media and transceivers, with these classes handling the
low-level details of connecting and disconnecting transceivers from media. Subclasses
of these base classes define the specifics of how transceivers connect to media and
how data is transmitted over the media.

:author: Alf O'Kenney

TODO:
- Make each symbol actually a tuple of symbol and duration, so that we can
    support modulation and frequency hopping.
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
from math import inf as _inf
from multiprocessing import Array, Event, Lock, Pipe, Process
from multiprocessing.connection import Connection, wait as conn_wait
from threading import Thread
from time import monotonic_ns
from typing import Callable, Sequence
from weakref import WeakValueDictionary

from .constants import (
    DEFAULT_NAME,
    NANOSECONDS_PER_SECOND,
    SPEED_OF_LIGHT,
    TIME_DILATION_FACTOR,
    ManagementMessages,
    Responses,
)
from .exceptions import (
    ConnectionError,
    ProcessNotRunningError,
    StopProcess,
    TransmissionComplete,
)
from ..space import Location, euclidean_distance


__all__ = ['Medium', 'Transceiver']

log = logging.getLogger(__name__)

Number = int | float
Symbol = Number | Callable

ConnRequest = namedtuple('ConnRequest', ['conn', 'location', 'create', 'kwargs'])


@lru_cache
def _dilate_time(real_time: int) -> int:
    return real_time * TIME_DILATION_FACTOR


@lru_cache
def _undilate_time(dilated_time: int) -> int:
    return dilated_time / TIME_DILATION_FACTOR


class Transmission:
    """A class for describing a transmission with respect to two :class:`Transceiver`
    objects."""

    def __init__(
        self,
        symbol: Symbol,
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
        self._stop_ns = start_ns + duration_ns
        self._attenuation = attenuation

    @cached_property
    def _symbol_fn(self) -> Callable:
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

        # We need to undilate the time to get the correct amplitude. We are sampling at a
        # lower rate than the transceivers would be in real life. That is, the difference
        # between the start time and the current time is TIME_DILATION_FACTOR times larger
        # than it would be in real life.
        actual_ns = _undilate_time(time_ns - self._start_ns)
        return self._symbol_fn(actual_ns) * self._attenuation


class ReceptionManager:
    """A class for managing incoming transmissions from a :class:`Medium`.
    :class:`Transceiver`s add :class:`Transmission`s to this object upon being notified by
    the :class:`Medium` that a transmission has been received. The :class:`Transceiver`
    then uses thsi object to measure the energy of the medium at a specified time. It is
    the responsibility of this class to ensure that the transceivers are reading the
    aggregated amplitude of all transmissions at this time.
    """

    def __init__(self):
        self._transmissions: set[Transmission] = set()

    @property
    def next_rx_ns(self) -> int:
        """Walk trhough the transmission that are on the way and return the earliest start
        time.
        """
        if not self._transmissions:
            return _inf

        # Find the transmission that will arrive (arrove?) first
        return min(tx._start_ns for tx in self._transmissions)

    def add_transmission(
        self,
        symbol: Callable | Number,
        start_ns: float,
        duration_ns: float,
        attenuation: float = 1.0,
    ):
        """Used by the :class:`Transceiver` to update the set of transmissions it is going
        to receive.

        :param symbol: Either a static amplitude or a function that takes the time since
            the start of the transmission as its input and returns the amplitude of the
            signal at that time.
        :param start_ns: The time in nanoseconds (in the :class:`Transceiver`'s clock) at
        which the transmission begins.
        :param duration_ns: The duration of the transmission (dilated / from the
            :class:`Transceiver` object's perspective) in nanoseconds.
        :param attenuation: The attenuation of the transmission once it reaches the
            :class:`Transceiver`.
        """
        self._transmissions.add(Transmission(symbol, start_ns, duration_ns, attenuation))

    def get_amplitude(self, time_ns: float) -> float:
        """Read the aggregated amplitude of all transmissions at a given time.

        :param time_ns: The time in nanoseconds at which to read the amplitude.

        :returns: The aggregated amplitude of all transmissions at the given time.
        """
        # Collect the summation of the current amplitudes of all of the transmissions.
        amplitude = 0.0

        # Work with a shallow copy so that we can remove transmissions from the original
        # set as we go.
        for tx in self._transmissions.copy():
            try:
                amplitude += tx.get_amplitude(time_ns)
            except TransmissionComplete:
                # The transmission has finished, so we must remove it from the set.
                self._transmissions.remove(tx)
            except Exception as e:
                # Leave the transmission in the set and log the error. The transmission
                # class itself will tell us when it's time to remove it.
                log.error(
                    f'{self}: an exception occurred while processing a transmission: {e}'
                )


class ConnDetails:
    """A container for the connection details between a :class:`Transceiver` and a
    :class:`Medium`.
    """

    def __init__(self, loc: Location, current_ns: int):
        """
        :param loc: The location of the :class:`Transceiver` relative to the
            :class:`Medium`.
        :param current_ns: The current time in nanoseconds.
        """
        self.location: Location = loc

        # The last time the medium received a signal from the transceiver and the last
        # time the medium sent a signal to the transceiver, respectively. Both values are
        # from the perspective of the medium and both are set to be the current time,
        # since the creation of a connection between the transceiver and the medium
        # implies that the medium has received a message from the transceiver and sent a
        # reply to the transceiver.
        self.last_rx_ns: float = current_ns
        self.last_tx_ns: float = current_ns


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

        # The dict of details about connected transceivers, and the lock used to ensure
        # thread safety.
        self._conn_details: dict[Location, ConnDetails] = {}
        self._conn_details_lock: Lock = Lock()

        # The connection over which the main process uses to communicated the addition or
        # removal of transceivers from the link. The client is used to forward connection
        # requests from Transceivers into the Medium process and the listener is monitored
        # within the process.
        conn_listener, conn_client = Pipe(duplex=False)
        self._connections_client: Connection = conn_client
        self._connections_listener: Connection = conn_listener

        # This event is used to stop both ourselves (the forked Process) and Thread
        # workers.
        self._stop_event: Event = Event()

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

    # region Multiprocessing- / IPC-related methods

    @property
    def _connections(self) -> tuple[Connection]:
        """Return a tuple of connections to the `Medium` for all of the connected
        `Transceiver`s.
        """
        with self._conn_details_lock:
            return tuple(self._conn_details)

    def run(self):
        log.info(f'Starting medium process ({self.pid})')

        while not self._stop_event.is_set():
            for conn in conn_wait((*self._connections, self._connections_listener)):
                if conn is self._connections_listener:
                    try:
                        self._process_connections()
                    except StopProcess:
                        # We've been told to stop, so we do so while also ensuring any
                        # threads are alerted to the fact that they should stop.
                        self._stop_event.set()
                        break
                else:
                    self._process_tx_event(conn)

        log.debug(f'{self}: shutting down')

    def _process_connections(self):
        """This method processes the connections `Connection` for updates to the set of
        connected transceivers and connects or disconnects them as appropriate.
        """
        while self._connections_listener.poll():
            try:
                conn_req = self._connections_listener.recv()
            except EOFError:
                log.debug(f'{self}: connection listener closed')
                raise StopProcess from None
            except Exception as e:
                log.error(f'{self}: connection listener error: {e}')
                continue

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

    def _add_connection(self, conn_req: ConnRequest):
        """Add a connection between a :class:`Transceiver` and this :class:`Medium` and
        return the details of this connection to the :class:`Transceiver`.

        :param creq: The connection request to use to configure the connection.
        """
        now = monotonic_ns()
        conn = conn_req.conn

        try:
            location = self._connect(conn_req)
        except Exception as e:
            log.error(f'{self}: connection failed: {e}')

            try:
                conn.send((Responses.ERROR, e))
            except Exception as e:
                log.error(f'{self}: connection error reply to transceiver failed: {e}')
            return

        new_conn_details = ConnDetails(location, now)

        with self._conn_details_lock:
            self._conn_details[conn] = new_conn_details

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
            with self._conn_details_lock:
                del self._conn_details[conn]
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
        with self._conn_details_lock:
            # Make sure to not just remove the connection, but also to close it.
            try:
                self._conn_details.pop(conn := conn_req.conn)
            except KeyError:
                log.info(f'{self}: connection not found: {conn}')
                return

            conn.close()

    def _process_tx_event(self, src_conn: Connection):
        """Process a transmission event for a :class:`Transceiver`.

        :param src_conn: The connection to the :class:`Transceiver` which is transmitting.
        """
        # Get the symbol and delta for this transmission.
        try:
            tx_details = src_conn.recv()
        except EOFError:
            # It's possible that the transceiver is dead without having been disconnected.
            # Given that the connection is closed, we should just remove it.
            log.info(
                f'{self}: connection to {src_conn} closed. Removing from connections.'
            )

            with self._conn_details_lock:
                try:
                    del self._conn_details[src_conn]
                except KeyError:
                    pass

            return
        except Exception as e:
            log.error(f'{self}: unexpected exception while receiving tx details: {e}')
            return

        try:
            symbol, ns_since_last_xcvr_tx, dilated_duration_ns = tx_details
        except (ValueError, TypeError):
            log.error(f'{self}: unexpected transmission details: {tx_details}')
            return

        with self._conn_details_lock:
            # Get the connection details for this transceiver.
            src_conn_details = self._conn_details[src_conn]

            # Use the delta and the last transmission time for this transceiver to
            # determine when this transmission was scheduled to be sent from our
            # perspective.
            local_tx_time_ns = src_conn_details.last_tx_ns + ns_since_last_xcvr_tx

            # Keep track of this new "last" time for the source transceiver.
            src_conn_details.last_tx_ns = local_tx_time_ns

            # Relay the symbol and the deltas to all other transceivers.
            for dest_conn, dest_conn_details in self._conn_details.items():
                if dest_conn is src_conn:
                    continue

                self._effectuate_transmission(
                    src_conn_details.location,
                    dest_conn,
                    dest_conn_details,
                    local_tx_time_ns,
                    dilated_duration_ns,
                    symbol,
                )

    def _effectuate_transmission(
        self,
        src_loc: Location,
        dest_conn: Connection,
        dest_details: ConnDetails,
        local_tx_time_ns: int,
        dilated_duration_ns: int,
        symbol: Symbol,
    ):
        # Determine the delta between the this transmission and the last time we
        # sent something to this destination.
        xcvr_rx_delta_ns = local_tx_time_ns - dest_details.last_rx_ns

        # Sort the locations to take advantage of the invertible nature of the
        # travel time calculation.
        # TODO:
        # Test if this is efficient for both small and large numbers of
        # transceivers as well as at the beginning of a simulation when there are
        # no cached travel times and when the simulation has been running for a
        # while and there are many cached travel times.
        dilated_prop_delay_ns = self._calculate_travel_time_ns(
            *sorted((src_loc, dest_details.location)),
            dilate=True,
        )

        try:
            dest_conn.send(
                (
                    symbol,
                    xcvr_rx_delta_ns,
                    dilated_prop_delay_ns,
                    dilated_duration_ns,
                    1,  # No attenuation for now.
                )
            )
        except Exception as e:
            log.error(
                (
                    f'{self}: unexpected exception while sending tx details to '
                    f'location={dest_details.location}: {e}'
                )
            )

            # Do not to update the last transmission time for this destination (since we
            # didn't actually send anything).
            return

        # Update the last time we sent something to this destination.
        dest_details.last_rx_ns = local_tx_time_ns

    @classmethod
    @lru_cache
    def _calculate_travel_time_ns(
        self, loc_1: Location, loc_2: Location, dilate: bool = True
    ) -> float:
        """Calculate the time (in nanoseconds) it takes for a signal to travel from one
        location to another in this medium."""
        real_ns = euclidean_distance(loc_1, loc_2) / self._medium_velocity_ns

        if dilate:
            return _dilate_time(real_ns)

        return real_ns

    # endregion

    # region Public methods

    def connect(self, conn: Connection, location: Location = None, **kwargs):
        """Communicate with the worker process to establish a connection with a
        :class:`Transceiver`.

        :param conn: our :class:`~multiprocessing.Connection` to the
            :class:`~multiprocessing.Pipe` for communicating with the :class:`Transceiver`
        :param location: the desired location of the :class:`Transceiver` in the medium.
            If not provided, the :class:`Transceiver` will be placed at an location
            determined by the medium.
        """
        self._connections_client.send(ConnRequest(conn, location, True, kwargs))

    def disconnect(self, conn: Connection, location: Location, **kwargs):
        """Communicate with the worker process to disconnect a :class:`Transceiver`.

        NOTE:
        At some point we may want to add a check to make sure that the transceiver
        disconnecting is the one that connected.

        :param location: the location of the :class:`Transceiver` in the medium.
        """
        self._connections_client.send(ConnRequest(conn, location, False, kwargs))

    def stop(self):
        """Let the worker thread break out of its processing loop in addition to the
        usual work done by :meth:`~multiprocessing.Process.terminate`.
        TODO:
        Think of some way to also disconnect all of the connected transceivers. Initially
        I was thinking that we could replace the location with the transceiver instances
        themselves, but I'll need to check that this is even possible for IPC."""
        self._stop_event.set()

        try:
            self._connections_client.send(ManagementMessages.CLOSE)
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
    """The base class for all transceivers. A transceiver is a device that can send and
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

        # Make sure all of the media have the same dimensionality. It doesn't make sense
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

        # Keep track of the next and last times associated with transfer of symbols
        # between the transceiver and the medium. The next times are used to determine
        # when to perform an action on our end, whereas the last times are used for
        # synchronization with the medium.
        self._next_rx_ns: int = None
        self._next_tx_ns: int = None
        self._last_medium_rx_ns: int = None
        self._last_medium_tx_ns: int = None

        self._tx_frame_symbols: list = None
        self._tx_frame_symbol_len: int = 0
        self._tx_index: int = 0

        # The base time delta between symbols nanoseconds. We start to receive symbols at
        # intervals of this value. The receiving delta could float up or down depending on
        # the algorithm used to detect the rate/phase of the incoming signal.
        self._base_delta_ns: int = NANOSECONDS_PER_SECOND // base_baud
        self._dilated_base_delta_ns: int = _dilate_time(self._base_delta_ns)

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

        # The connection over which the transceiver communicates with the medium.
        self._conn_2_med: Connection = None

        # The connection of the medium to the transceiver. We will only use this to send
        # a disconnect message to the medium, which is using this Connection object as a
        # key in its connections dictionary.
        self._conn_in_medium: Connection = None

        # An event to let both the main Process and any worker threads know that it's
        # time to shut down.
        self._stop_event: Event = Event()

        # The pipe over which the bottom two layers of the OSI model communicate.
        l2_osi_conn, l1_osi_conn = Pipe(duplex=True)
        self.l2_osi_conn: Connection = l2_osi_conn
        self._l1_osi_conn: Connection = l1_osi_conn

        self._rx_manager: ReceptionManager = ReceptionManager()
        self._rx_manager_lock: Lock = Lock()

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
    def _translate_frame(self, frame: tuple[int]):
        """Translate the frame from L2 into a sequence of symbols as they will appear in
        the medium.

        It is up to the subclasses to determine how to do this (e.g., encoding,
        modulating), and the result will be sent to the medium. For example, a
        :class:`Transceiver` which uses Manchester encoding might each bit and convert it
        into the two symbols needed to represent it.
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

    # region Multiprocessing-related / IPC-related methods

    def run(self):
        log.info(f'Starting transceiver process ({self.pid})')

        while not self._stop_event.is_set():
            # Wait for a connection update.
            new_conn = self._connections_listener.recv()
            if new_conn is ManagementMessages.CLOSE:
                break

            if new_conn is not None:
                # Start up a new thread which will use these connections to communicate
                # data, process it and relay the result to the next layer up.
                self._conn_2_med = new_conn

                # We'll consider this as being a transmission to the medium (more of a
                # meta transmission, but it starts the clock that we will use to compare
                # against future comms).
                self._last_medium_tx_ns = monotonic_ns()
                thread = Thread(target=self._monitor_medium)
                thread.start()
            else:
                # Stop the thread which is currently receiving bits.
                self._conn_2_med.send(ManagementMessages.CLOSE)
                thread.join()

                self._conn_2_med.close()
                self._conn_2_med = None
                self._medium = None
                self._last_medium_tx_ns = None

        # If we're exiting the loop, we need to make sure that the thread is joined.
        if self._conn_2_med is not None:
            self._conn_2_med.send(ManagementMessages.CLOSE)
            thread.join()

        log.debug(f'{self}: shutting down')

    def _monitor_medium(self):
        """This method has several responsibilities:
            1. Wait for symbols from the medium and add them to the set of transmissions
                to be processed.
            2. Wait for frames from the next layer up and ensure that they are sent to
                the medium, symbol by symbol.
            3. Reading the medium's intensity either at prescribed intervals or when it
                goes from being idle to being active.
            4. Sending completed frames to the next layer up.

        These events can be categorized into two groups: those which are expected and
        occur after a defined delta, and those which are unexpected and arrive randomly
        from either the medium or the next layer up.

        It will be run in a :class:`~Thread` whose lifetime is only as long as the
        lifetime of the connection to the medium. That is, the :class:`~Thread` will be
        started when the connection is first made and will be stopped when the connection
        is terminated.
        """
        conns_to_monitor = (self._conn_2_med, self._l1_osi_conn)

        while not self._stop_event.is_set():
            # Walk through the knowns.
            self._process_receptions()
            self._process_transmissions()

            try:
                next_event_ns = min(
                    event for event in (self._next_tx_ns, self._next_rx_ns) if event
                )
            except ValueError:
                # There are no known events, so we wait indefinitely for something to come
                # in from either the medium or the next layer up.
                timeout = None
            else:
                # Get the delta until the next event.
                timeout = next_event_ns - monotonic_ns()

            # Until that event occurs, wait for something to arrive from either the medium
            # or the next layer up.
            for conn in conn_wait(conns_to_monitor, timeout=timeout):
                if conn is self._conn_2_med:
                    # We've received a transmission from the medium.
                    self._process_new_medium_reception()
                elif conn is self._l1_osi_conn:
                    # We've received a new transmission from the next layer up. Collect it
                    # and prepare to send it.
                    self._process_new_l2_frame()

    def _process_receptions(self):
        """
        It's the responsibility of the subclass to determine what to do with the observed
        amplitude (e.g., how to manipulate/decode it, where to put it). Addtionally, the
        subclass must give us the amount of nanoseconds to wait before sampling the
        :class:`Medium` again, with a value of -1 indicating that we should only be
        informed of the next reception event (simulating the monitoring of the "active"
        bit).
        """
        current_ns = monotonic_ns()

        # Get the next time that a known reception event will occur.
        # NOTE:
        # `ReceptionMananger.next_rx_ns` is a property which performs a calculation
        # based on the various transmissions observed by this Python object. It is
        # therefor more efficient to only run this calculation _after_ we've
        # determined that we weren't in an active listening mode (e.g., though the use
        # of `or` short-circuiting).
        sample_ns = self._next_rx_ns or self._rx_manager.next_rx_ns

        if not sample_ns or sample_ns > current_ns:
            # No value: there are no known reception events.
            # Active listening mode: it's not yet time to sample the ongoing transmission.
            # Passive listening mode: the incoming reception has not yet arrived.
            return

        # Active listening mode:
        #   the next symbol in the ongoing transmission has arrived and it's time to
        #   sample it.
        # Passive listening mode:
        #   the medium has become active and it's time to sample it.
        amplitude = self._rx_manager.get_amplitude(sample_ns)

        # For whatever reason, we just read an amplitude from the medium. We need to
        # process it and determine when the next sample should be taken.
        rx_sample_delta_ns = self._process_rx_amplitude(amplitude)

        if rx_sample_delta_ns == -1:
            # We're finished processing a transmission, so we switch back to passive
            # listening mode.
            self._next_rx_ns = None
            return

        if rx_sample_delta_ns is None:
            # If the subclass didn't give us a next reception time, we'll just default to
            # half of the base delta, in keeping with the Nyquist-Shannon sampling
            # theorem.
            rx_sample_delta_ns = self._base_delta_ns / 2

        # Configure the next reception time, accounting for time dilation.
        self._next_rx_ns = sample_ns + _dilate_time(rx_sample_delta_ns)

    def _process_transmissions(self):
        if not (next_tx_ns := self._next_tx_ns) or next_tx_ns > monotonic_ns():
            # Either we're not currently transmitting or it's not yet time to send the
            # next symbol.
            return

        try:
            symbol = self._tx_frame_symbols[self._tx_index]
        except IndexError:
            # We've gone past the end of the frame (implying no collisions), which means
            # we're done sending the frame.
            self._tx_frame_symbols = None
            self._tx_frame_symbol_len = None
            self._tx_index = 0
            self._next_tx_ns = None
            return

        # We need to let the medium the delta between the start of the last communication
        # and when this communication was scheduled to start.
        # NOTE:
        # In the trivial, in-frame case, this is just the base delta. However, if we're
        # beginning a new frame, we need to account for the time between frames.
        ns_since_last_tx = next_tx_ns - self._last_medium_tx_ns

        dilated_duration_ns = self._dilated_base_delta_ns

        try:
            # Send the symbol to the medium, letting it know how many nanoseconds have
            # passed and how long the symbol should be.
            self._conn_2_med.send((symbol, ns_since_last_tx, dilated_duration_ns))
        except Exception as e:
            log.error(f'{self}: Error sending symbol: {e}')

        # Store the time that we sent the symbol to the medium such that we may reference
        # it when sending the next symbol.
        self._last_medium_tx_ns = next_tx_ns

        # Regardless of whether or not the transmission was successful, we need to
        # increment the index and update the next transmission time with proper
        # inter-symbol spacing.
        self._tx_index += 1
        self._next_tx_ns += dilated_duration_ns

    def _process_new_medium_reception(self):
        """We've received a new transmission from the medium. We need to add it to the
        set of transmissions to be processed.
        """
        try:
            symbol_details = self._conn_2_med.recv()
        except Exception:
            log.error(f'{self}: Error receiving symbol details from medium')
            return

        try:
            (
                symbol,
                delta_ns,
                dilated_prop_delay_ns,
                dilated_duration_ns,
                attenuation,
            ) = symbol_details
        except ValueError:
            log.error(
                f'{self}: Received invalid symbol details from medium: {symbol_details!r}'
            )
            return

        # Use the medium's view of the time since it last sent us a symbol to determine
        # when the symbol began transmission from this Python object's perspective.
        local_rx_time_ns = self._last_medium_rx_ns + delta_ns

        with self._rx_manager_lock:
            self._rx_manager.add_transmission(
                symbol,
                local_rx_time_ns + dilated_prop_delay_ns,
                dilated_duration_ns,
                attenuation,
            )

        # Store the time that we received the symbol from the medium object
        # NOTE:
        # This is not when the symbol first starts being received by the actual
        # transceiver (which would include a propagation delay). This is just used to
        # track comms between the Python objects.
        self._last_medium_rx_ns = local_rx_time_ns

    def _process_new_l2_frame(self):
        """We've received a new frame from the next layer up. We need to add it to the
        set of frames to be processed.
        """
        try:
            frame = self._l1_osi_conn.recv()
        except Exception:
            log.error(f'{self}: Error receiving frame from next layer up')
            return

        # Convert the frame into the symbols that will be put on the medium. This should
        # be some combination of modulation and encoding.
        # TODO:
        # Make each symbol actually a tuple of symbol and duration, so that we can
        # support modulation and frequency hopping.
        try:
            self._tx_frame_symbols = self._translate_frame(frame)
        except Exception as e:
            log.error(f'{self}: Error translating frame: {e}')
            return

        self._tx_frame_symbol_len = len(self._tx_frame_symbols)
        self._tx_index = 0
        self._next_tx_ns = monotonic_ns()

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

        # The subclass is okay with the connection, so create a pipe for transmission and
        # receptions with respect to link (i.e., medium <-> transceiver).
        medium_conn, xcvr_conn = Pipe(duplex=True)

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

        # Configure ourselves with the other ends of the pipe.
        # NOTE:
        # Because this method is being accessed outside of the process, we need to use
        # our own connection pipe to communicate the change to the process.
        try:
            self._connections_client.send(xcvr_conn)
        except Exception as e:
            raise ConnectionError(
                f'Error sending connection pipe to {self!r} process: {e}'
            )

        # TODO:
        # Get ack from process that it has received the connection pipe and is ready to
        # work.

        # The medium has acknowledged the connection and our worker process is now has a
        # thread running to communicate iwth the medium, meaning that we are now connected
        # and can set our location and other metadata.
        self.location = location

        # Store the Medium's connection for use in the disconnect method.
        self._conn_in_medium = medium_conn

        # Keep track of the medium so we can disconnect from it later or prevent the
        # changes seen at the top of this method.
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
        medium.disconnect(self._conn_in_medium)

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
