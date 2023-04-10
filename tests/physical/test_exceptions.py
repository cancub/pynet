from unittest import TestCase, main

from pynet.physical.exceptions import TransmissionComplete


class TestExceptions(TestCase):
    def test_transmission_complete(self):
        self.assertEqual(
            str(TransmissionComplete()), 'The symbol has finished being received'
        )


if __name__ == '__main__':
    main()
