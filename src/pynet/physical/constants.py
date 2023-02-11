from enum import IntEnum

# Meters per second in a vacuum
SPEED_OF_LIGHT = 299_792_458

CLOSE_MSG = object()


class Responses(IntEnum):
    """Responses from the server."""

    OK = 0
    ERROR = 1
    CLOSE = 2
