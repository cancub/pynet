from functools import lru_cache
from typing import Iterable

number = int | float


class RingBuffer:
    def __init__(self, size: int, initializer: Iterable = None, default: number = 0):
        self._size = size
        self._default = default
        self._index = 0

        try:
            self._data = list(initializer)
        except TypeError:
            self._data = [default] * size
        else:
            # Fill in the remainder of the buffer with the default value.
            self._data += [default] * (size - len(self._data))

    @lru_cache
    def _get_effective_index(self, index: int, offset: int = 0):
        """Get the effective index of the buffer, taking into account the offset and
        the size of the buffer.

        Cache the result to avoid unnecessary calculations.

        :param index: The index to start at.
        :param offset: The offset to add to the index."""
        return (index + offset) % self._size

    def append(self, item: number):
        """Append an item to the buffer, overwriting the oldest item."""
        self._data[self._index] = item
        self._index = self._get_effective_index(self._index, 1)

    def shift(self):
        """Shift the buffer by one, adding the default value to the end."""
        self.append(self._default)

    def overlay(self, items: Iterable[number], offset: int = 0):
        """Overlay a list of items onto the buffer, addind to the existing items and
        starting at the given offset.

        :param items: The items to overlay.
        :param offset: The offset to start at.
        """
        if (overlay_len := len(items)) > self._size:
            raise ValueError(
                f'Cannot overlay {overlay_len} items onto a buffer of size {self._size}.'
            )

        # We also want to catch the situation where the offset plus the length of the
        # items is greater than the size of the buffer, which would cause spillover.
        if overlay_len + offset > self._size:
            raise ValueError(
                f'Cannot overlay {overlay_len} items at offset {offset} onto a buffer of '
                f'size {self._size}. This would overflow beyond the intended range.'
            )

        start = self._index + offset
        for i, item in enumerate(items):
            self._data[self._get_effective_index(start, i)] += item

    def get_current(self) -> number:
        return self._data[self._index]

    def popleft(self) -> number:
        current = self.get_current()
        self.shift()
        return current
