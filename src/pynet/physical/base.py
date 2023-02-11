"""
Base classes for physical media and transceivers, with these classes handling the
low-level details of connecting and disconnecting transceivers from media. Subclasses
of these base classes define the specifics of how transceivers connect to media and
how data is transmitted over the media.

:author: Alf O'Kenney

TODO:
- use multiprocessing.connection.wait to collect data from all transceivers
    - make connections full duplex
- make Medium a subclass of Queue?
    - need to investigate if this is any simpler than the current implementation and if
        it would even be feasible
    - it would be nice because it would be one less Process to manage
    - the main thing here would be sending copies of inputs to all nodes, with additonal
        metadata.
    - we would need to track which nodes have received which data, and only send data to
        nodes that haven't received it yet.
    - input gets duplicated on put()s, with metadata added for specific receivers
    - get() requires a location argument, and returns the copy of the data that is
        intended for that location
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
from multiprocessing import Array, Event, Lock, Pipe, Process, Queue, Value
from threading import Thread
from typing import TYPE_CHECKING, Any, Sequence
from weakref import WeakValueDictionary

from .constants import CLOSE_MSG, Responses
from .exceptions import ConnectionError, NoMediumError, ProcessNotRunningError

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

__all__ = ['Medium', 'Transceiver']

log = logging.getLogger(__name__)

# A location in 1D or 3D space.
Location = float | tuple[float, float, float]

# It's not elegant, but it likely gives users enough space to describe what they want.
DEFAULT_NAME = b' ' * 64


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

    TODO:
    - give transmissions a time limit. Do not let them sit around forever.
        - we can probably do this if we implement the Medium as a subclass of Queue by
            only providing values on get() that would have been heard from the location
            prior to connecting to the Medium.
        - this might be moot though, since in practice the odds of this happening are
            probably very low.
    """

    # Keep track of all instances so we can shut down their respective processes at exit
    _instances = WeakValueDictionary()

    # The number of dimensions in which the medium exists. For example, coaxial cable is
    # 1D, while free space/air is 3D.
    _dimensionality: int

    # region Dunders

    def __init_subclass__(cls, dimensionality: int, **kwargs):
        super().__init_subclass__()
        if dimensionality not in (1, 3):
            raise ValueError('`dimensionality` must be 1 or 3')

        cls._dimensionality = dimensionality

    def __new__(cls, name: str, *args, **kwargs):
        try:
            return cls._instances[cls, name]
        except KeyError:
            pass
        obj = super().__new__(cls)
        cls._instances[cls, name] = obj
        return obj

    def __init__(self, name: str, auto_start: bool = True):
        super().__init__()

        self.name = name

        self._connection_thread: Thread = None

        # The queue on which data is received from Transceivers.
        # NOTE: Monitored by this Process.
        self._tx_ingress_queue: Queue = Queue()

        # Connections on which each of the connected Transceivers receive data and the
        # lock used to ensure thread-safety when accessing the group.
        self._receivers: dict[Location, Connection] = {}
        self._receivers_lock: Lock = Lock()

        # The Queue monitored by the worker thread for updates to state of connected
        # Transceivers.
        self._connection_queue: Queue = Queue()

        # This event is used to stop both ourselves (the forked Process) and Thread
        # workers.
        self._stop_event: Event = Event()

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

    @abstractmethod
    def _process_transmission(self, data: Any, src_loc: Location, dst_loc: Location):
        """Process a bit of data sent by a :class:`Transceiver` instance."""
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
        self._connection_thread = Thread(
            target=self._monitor_connections,
            args=(
                self._connection_queue,
                self._receivers,
                self._receivers_lock,
                self._stop_event,
            ),
        )
        self._connection_thread.start()

        while not self._stop_event.is_set():
            self._process_medium()

        self._connection_thread.join()

        log.debug(f'{self}: shutting down')

    def _monitor_connections(
        self,
        connection_queue: Queue,
        receivers: dict,
        receivers_lock: Lock,
        stop_event: Event,
    ):
        """To be run in a separate thread, this method watches the connections queue for
        updates to the set of connected transceivers and connects or disconnects them as
        appropriate."""
        log.info(f'Starting connection worker thread ({self.pid})')

        while not stop_event.is_set():
            # Wait for a connection update.
            conn_details = connection_queue.get()
            if conn_details is CLOSE_MSG:
                break

            try:
                conn, location, kwargs = conn_details
            except TypeError:
                log.error(f'{self}: unexpected connection format: {conn_details}')
                continue

            if not location:
                # When no location is provided, we assume that the connection is being
                # established.
                try:
                    location = self._connect(**kwargs)
                except Exception as e:
                    log.error(f'{self}: connection failed: {e}')
                    conn.send((Responses.ERROR, e))
                    continue

                with receivers_lock:
                    receivers[location] = conn

                # Send the location back to the transceiver so it can be stored,
                # acknowledging the connection in the process.
                conn.send((Responses.OK, location))
            else:
                # TODO:
                # Perform some sanity checks to make sure that the connection is being
                # closed by the correct transceiver.
                try:
                    self._disconnect(location)
                except Exception as e:
                    log.error(
                        f'{self}: subclass disconnection failed: {e}. Continuing with '
                        'removal.'
                    )

                with receivers_lock:
                    conn = receivers.pop(location)

                # Make sure to close the connection to the transceiver.
                conn.close()

        log.info(f'Connection worker thread exiting ({self.pid})')

    def _process_medium(self):
        transmission = self._tx_ingress_queue.get()
        if transmission is CLOSE_MSG:
            return

        try:
            data, src_loc = transmission
        except (ValueError, TypeError):
            log.error(
                f'{self}: invalid transmission received ({transmission}). Format must be (location, data)'
            )
            return

        # Make sure to keep the set of receivers static while we're broadcasting.
        with self._receivers_lock:
            # We only need to send to the transceivers that are not co-located with
            # the transmission.
            # NOTE:
            # The location of the receivers is important for determining what each
            # receiver should receive and when. Consider the extreme case of a WiFi
            # setup where one receiver is in what has been defined as a faraday cage.
            for dest_loc, dest_conn in self._receivers.items():
                if dest_loc == src_loc:
                    continue

                # Don't let an exception in _process_transmission() kill the thread.
                try:
                    data = self._process_transmission(data, src_loc, dest_loc)
                except Exception as e:
                    log.error(
                        f'{self}: error processing transmission ({data=}, {src_loc=}, {dest_loc=}): {e}'
                    )
                    continue

                try:
                    dest_conn.send(data)
                except Exception as e:
                    log.error(
                        f'{self}: error sending {data=} to receiver at {dest_loc=}: {e}'
                    )

    # endregion

    # region Public methods

    def transmit(self, data: Any, location: Location):
        """Send data over the medium from a specific location."""
        self._tx_ingress_queue.put((data, location))

    def stop(self):
        """Let the worker thread break out of its processing loop in addition to the
        usual work done by :meth:`~multiprocessing.Process.terminate`.
        TODO:
        Think of some way to also disconnect all of the connected transceivers. Initially
        I was thinking that we could replace the location with the transceiver instances
        themselves, but I'll need to check that this is even possible for IPC."""
        self._stop_event.set()

        for q in (self._connection_queue, self._tx_ingress_queue):
            try:
                q.put(CLOSE_MSG)
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

    # region Dunders

    def __init_subclass__(cls, supported_media: Sequence[Medium]) -> None:
        bad_media = [m for m in supported_media if not issubclass(m, Medium)]
        if bad_media:
            raise TypeError(f'{bad_media} are not subclasses of {Medium}')

        # Make sure all of the media have the same dimesnionality. It doesn't make sense
        # to have a transceiver that can connect to one type of medium that only exists
        # in 1D (e.g., coax) and another that exists in 3D (air).
        if len(set(m._dimensionality for m in supported_media)) != 1:
            raise ValueError('`supported_media` must have the same number of dimensions.')

        cls._supported_media = supported_media

    def __new__(cls, name: str, *args, **kwargs):
        try:
            return cls._instances[cls, name]
        except KeyError:
            pass
        obj = super().__new__(cls)
        cls._instances[cls, name] = obj
        return obj

    def __init__(self, name: str, auto_start: bool = True):
        super().__init__()

        self.name: str = name
        self._medium: Medium = None

        # The type of the shared memory used for the location depends on the number of
        # axes that the location is defined over.
        if (axes_count := self._supported_media[0]._dimensionality) == 1:
            self._location: Value = Value('f', 0.0)
        else:
            self._location: Array = Array('f', [0.0] * axes_count)

        # Store the name as a value so that it can be accessed within the process.
        self._medium_name: Array = Array('c', DEFAULT_NAME)

        # The shared objects over which data is transmitted and received, respectively.
        self._tx_queue: Queue = None
        self._rx_conn: Connection = None

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
    def _process_incoming_data(self, data: Any):
        """Process data received from the medium"""
        raise NotImplementedError

    @abstractmethod
    def _process_outgoing_data(self, data: Any):
        """Process data to be sent to the medium"""
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
        current_phy_listener: Connection = None

        while not self._stop_event.is_set():
            # Wait for a connection update.
            new_phy_listener = self._connections_listener.recv()
            if new_phy_listener is CLOSE_MSG:
                break

            if new_phy_listener is not None:
                # Start up a new thread which will use this connection to receive data,
                # process it and relay the result to the next layer up.
                thread_stop_event.clear()
                thread = Thread(
                    target=self._monitor_medium,
                    args=(new_phy_listener, self._osi_client, thread_stop_event),
                )
                thread.start()

                # Store the new connection such that we may send a close message on
                # disconnection, thereby letting the thread unblock.
                current_phy_listener = new_phy_listener
            else:
                # Stop the thread which is currently receiving bits.
                thread_stop_event.set()
                current_phy_listener.send(CLOSE_MSG)
                thread.join()

                current_phy_listener = None
                self._medium = None

        log.debug(f'{self}: shutting down')

    def _monitor_medium(
        self, phy_listener: Connection, osi_client: Connection, stop_event: Event
    ):
        """Watch the medium for incoming data. This method will be run in a Thread whose
        lifetime is only as long as the lifetime of the connection to the medium. That is,
        the thread will be started when the connection is first made and will be stopped
        when the connection is terminated."""
        while not stop_event.is_set():
            try:
                data = phy_listener.recv()
            except Exception as e:
                log.error(f'{self}: Error receiving data from medium: {e}')
                continue

            if data is CLOSE_MSG:
                break

            try:
                data = self._process_incoming_data(data)
            except Exception as e:
                log.error(f'{self}: Error processing {data=}: {e}')
                continue

            try:
                osi_client.send(data)
            except Exception as e:
                log.error(f'{self}: Error sending {data=} to next layer up: {e}')

        # We've exited the loop. This means our work is done, and we must close the rx
        # link on our end.
        phy_listener.close()

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

        # The subclass is okay with the connection, so create a pipe for the rx link
        # (i.e., medium -> transceiver).
        phy_listener, phy_client = Pipe(duplex=False)

        # Give one end to the medium, the reception of which will trigger its connection
        # logic.
        # NOTE:
        # The medium is the client for the rx link. That is, the medium (client) will be
        # relaying data to the transceiver (listener).
        try:
            new_medium._connection_queue.put((phy_client, None, kwargs))
        except Exception as e:
            raise ConnectionError(
                f'Error sending connection request to {new_medium!r}: {e}'
            )

        # Wait for the medium to acknowledge the connection with a message containing our
        # location.
        try:
            response = phy_listener.recv()
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

        # Configure ourselves with the other end.
        # NOTE:
        # Because this method is being accessed outside of the process, we need to use
        # our own connection pipe to communicate the change to the process.
        self._connections_client.send(phy_listener)

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
        medium._connection_queue.put((None, self.location, None))

        # Finally, alert the process to the fact that we are disconnecting.
        # NOTE:
        # Because this method is being accessed outside of the process, we need to use
        # our own connection to communicate the change to the process.
        self._connections_client.send(None)

        self._medium = None
        self._medium_name.value = DEFAULT_NAME

    def transmit(self, data: Any):
        """Allow the next layer up to send data over the medium"""
        if not self.is_alive():
            raise ProcessNotRunningError(
                'Transceiver must be running before transmitting data'
            )
        if not self._medium:
            raise NoMediumError('Cannot transmit data without a medium')

        data = self._process_outgoing_data(data)
        self._medium.transmit(data, self.location)

    def stop(self):
        """Gracefull stop the transceiver process"""
        # Tell the process to get rid of any existing connection to a medium.
        self.disconnect()

        # # Let the process know that it should stop.
        self._stop_event.set()
        try:
            self._connections_client.send(CLOSE_MSG)
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
