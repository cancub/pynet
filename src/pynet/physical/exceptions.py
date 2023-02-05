class LocationError(Exception):
    pass


class NoSuitableLocationsError(LocationError):
    pass


class InvalidLocationError(LocationError):
    pass
