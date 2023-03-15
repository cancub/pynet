class LocationError(Exception):
    pass


class NoSuitableLocationsError(LocationError):
    pass


class InvalidLocationError(LocationError):
    pass


class ProcessNotRunningError(Exception):
    pass


class ConnectionError(Exception):
    pass
