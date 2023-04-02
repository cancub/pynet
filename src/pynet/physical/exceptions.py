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


class TransmissionException(Exception):
    pass


class TransmissionComplete(TransmissionException):
    def __str__(self):
        return 'The symbol has finished being received'
