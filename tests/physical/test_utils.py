#! /usr/bin/env python
from unittest import TestCase, main

from pynet.physical.utils import RingBuffer


class RingBufferTestCase(TestCase):
    def setUp(self):
        super().setUp()

        self.buffer_len = 5
        self.buffer_default = 0
        self.buffer = RingBuffer(self.buffer_len, default=self.buffer_default)

    def test_init(self):
        # Initialize with a list of 1s that's not as big as the full buffer size.
        ones = [1] * 3
        buffer = RingBuffer(
            self.buffer_len, initializer=ones, default=self.buffer_default
        )
        self.assertEqual(self.buffer._size, 5)
        self.assertEqual(buffer._data, ones + [0] * 2)
        self.assertEqual(self.buffer._default, 0)
        self.assertEqual(self.buffer._index, 0)

    def test_get_effective_index(self):
        for lap in (1, 2):
            for i in range(self.buffer_len):
                self.assertEqual(
                    self.buffer._get_effective_index(i + self.buffer_len * lap), i
                )

    def test_append(self):
        for i in range(self.buffer_len * 2):
            self.buffer.append(i)

            # The (new) oldest value is the value that was appended.
            previous_index = self.buffer._get_effective_index(self.buffer._index - 1)
            self.assertEqual(self.buffer._data[previous_index], i)

    def test_shift(self):
        for i in range(self.buffer_len * 2):
            self.buffer.shift()

            # The (new) oldest value is the default value.
            previous_index = self.buffer._get_effective_index(self.buffer._index - 1)
            self.assertEqual(self.buffer._data[previous_index], self.buffer_default)

    def test_overlay(self):
        # Overlay the original buffer with a list of 1s.
        ones_overlay = [1] * self.buffer_len
        self.buffer.overlay(ones_overlay)

        # Given that the buffer waws orignially full of zeroes, the buffer is now just 1s.
        self.assertEqual(self.buffer._data, ones_overlay)

        # Move the index up one to test an overlay that wraps around.
        self.buffer._index += 1

        # Do a partial overlay with an offset so that we wrap around
        self.buffer.overlay([1] * 3, offset=2)
        self.assertEqual(self.buffer._data, [2, 1, 1, 2, 2])

    def test_overlay_too_big(self):
        size = self.buffer_len + 1
        with self.assertRaisesRegex(
            ValueError,
            f'Cannot overlay {size} items onto a buffer of size {self.buffer._size}.',
        ):
            self.buffer.overlay([1] * size)

    def test_overlay_offset_too_big(self):
        offset = self.buffer_len + 2
        data = [1] * (self.buffer_len - 2)
        with self.assertRaisesRegex(
            ValueError,
            (
                f'Cannot overlay {len(data)} items at offset {offset} onto a buffer of '
                f'size {self.buffer_len}. This would overflow beyond the intended range.'
            ),
        ):
            self.buffer.overlay(data, offset=offset)

    def test_get_current(self):
        # The current value is the value at the oldest index.
        for i in range(self.buffer_len * 2):
            # We start with a buffer full of zeroes.
            oldest_value = max(0, i - self.buffer_len + 1)
            self.buffer.append(i)
            self.assertEqual(self.buffer.get_current(), oldest_value)

    def test_popleft(self):
        # Initialize with a range.
        buffer = RingBuffer(self.buffer_len, initializer=range(1, self.buffer_len + 1))

        for i in range(self.buffer_len):
            self.assertEqual(buffer.popleft(), i + 1)

            # The newest value is the default value.
            newest_index = buffer._get_effective_index(buffer._index + buffer._size - 1)
            self.assertEqual(buffer._data[newest_index], buffer._default)


if __name__ == '__main__':
    main()
