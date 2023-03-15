from enum import Enum

# Meters per second in a vacuum
SPEED_OF_LIGHT = 299_792_458

NANOSECONDS_PER_SECOND = 1_000_000_000

# A value governing how many units of time will actually pass in the simulation for
# every unit of time that passes in the real world.

# In an ideal world, we would be able to operate at the same speed as the actual hardware
# we're simulating. Unfortunately, we're not just limited by the speed the processor, but
# also by the time delay introduced by performing all the data-processing operations in
# both the Medium and the Tranceiver objects. Ultimately, though, the goal is not to
# reproduce networks whose characteristics are indestinguisable from real-world networks,
# and which can be used for real-world applications. Rather, the goal is to provide a tool
# that can be used to explore the effects of different network topologies and
# configurations on the performance of networks. Thus, the time dilation factor is a way
# to put the network in a sort of slow motion in order to accurately reproduce these
# effects.
TIME_DILATION_FACTOR = 1_000

# It's not elegant, but it likely gives users enough space to describe what they want.
DEFAULT_NAME = b' ' * 64


class Responses(Enum):
    """Responses from the medium to the transceiver."""

    OK = 'ok'
    ERROR = 'error'


class CommsType(Enum):
    """Messages related to data communications between the transceiver and the medium."""

    TRANSMIT = 'tx'
    RECEIVE = 'rx'


class ManagementMessages(Enum):
    """Messages related to management of objects."""

    CLOSE = 'close'
