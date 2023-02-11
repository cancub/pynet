#!/usr/bin/env python
from __future__ import annotations

import logging
from itertools import product
from unittest import TestCase, main
from unittest.mock import Mock, patch
from contextlib import ExitStack

from pynet.physical.base import Medium, Transceiver
from pynet.testing import LogTestingMixin, ProcessBuilderMixin

log = logging.getLogger(__name__)

TESTING_TARGET = 'pynet.testing'


class TestLogTestingMixin(TestCase):
    def setUp(self):
        super().setUp()

        class MixinTester(TestCase, LogTestingMixin):
            pass

        self.mixin_tester = MixinTester()

    def _run_logging_test(
        self,
        regex: bool,
        msgs_args: list[str | list[str]],
        in_logs: bool,
        log_str: str = None,
    ):
        not_str = '' if in_logs else 'Not'
        assertion = getattr(self.mixin_tester, f'assert{not_str}InLogs')

        for level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]:
            for msgs in msgs_args:
                with ExitStack() as stack:
                    stack.enter_context(
                        self.subTest(level=level, is_list=isinstance(msgs, list))
                    )
                    stack.enter_context(assertion(__name__, level, msgs, regex=regex))

                    if log_str:
                        log.log(level, log_str)

    def test_assert_in_logs_not_regex(self):
        self._run_logging_test(
            regex=False,
            msgs_args=['test', ['tes', 'est']],
            in_logs=True,
            log_str='test',
        )

    def test_assert_in_logs_with_regex(self):
        self._run_logging_test(
            regex=True,
            msgs_args=['tes.', [r't\w{2}t', 'te.t']],
            in_logs=True,
            log_str='test',
        )

    def test_assert_not_in_logs_not_regex(self):
        for log_str in [None, 'test']:
            self._run_logging_test(
                regex=False,
                msgs_args=['foo', ['bar', 'baz']],
                in_logs=False,
                log_str=log_str,
            )

    def test_assert_not_in_logs_with_regex(self):
        for log_str in [None, 'test']:
            self._run_logging_test(
                regex=True,
                msgs_args=['f.o', [r't\w{3}t', r't\dst']],
                in_logs=False,
                log_str=log_str,
            )

    def test_assert_in_logs_but_nothing_logged(self):
        with self.assertRaises(AssertionError):
            with self.mixin_tester.assertInLogs(__name__, logging.DEBUG, 'test'):
                pass


class MockMedium(Medium, dimensionality=1):
    """A helper class for testing the :class:`TestProcessBuilderMixin` mixin."""

    def _connect(self, args, **kwargs):
        pass

    def _disconnect(self, args, **kwargs):
        pass

    def _process_transmission(self, args, **kwargs):
        pass


class MockTransceiver(Transceiver, supported_media=[MockMedium]):
    """A helper class for testing the :class:`TestProcessBuilderMixin` mixin."""

    location: int = None

    def _connect(self, args, **kwargs):
        pass

    def _disconnect(self, args, **kwargs):
        pass

    def _process_incoming_data(self, args, **kwargs):
        pass

    def _process_outgoing_data(self, args, **kwargs):
        pass


class TestProcessBuilderMixin(TestCase):
    @patch.object(MockMedium, 'start')
    def test_build_medium(self, start_mock):
        class MixinTester(TestCase, ProcessBuilderMixin):
            medium_cls = MockMedium

        self.medium_tester = MixinTester()

        test_specs = product(
            # name
            (None, 'test'),
            # mocked
            (True, False),
            # is_alive
            (None, True, False),
            # auto_start
            (True, False),
        )

        for name, mocked, is_alive, auto_start in test_specs:
            auto_gen_name = f'test_{len(Medium._instances)}'

            medium = self.medium_tester.build_medium(
                name=name, mocked=mocked, is_alive=is_alive, auto_start=auto_start
            )

            with self.subTest(
                name=name, mocked=mocked, is_alive=is_alive, auto_start=auto_start
            ):
                self.assertIsInstance(medium, Medium)

                # Checks dependent on `mocked`
                self.assertEqual(mocked, isinstance(medium, Mock))

                if not mocked and auto_start:
                    start_mock.assert_called_once()
                else:
                    start_mock.assert_not_called()

                self.assertEqual(mocked, isinstance(medium._connection_queue, Mock))

                # Checks independent of `mocked`
                if name is not None:
                    self.assertEqual(name, medium.name)
                else:
                    self.assertEqual(auto_gen_name, medium.name)

                if is_alive is not None:
                    self.assertEqual(is_alive, medium.is_alive())

            start_mock.reset_mock()

    @patch.object(MockTransceiver, 'start')
    def test_build_xcvr(self, start_mock):
        class MixinTester(TestCase, ProcessBuilderMixin):
            xcvr_cls = MockTransceiver

        self.xcvr_tester = MixinTester()

        test_specs = product(
            # name
            (None, 'test'),
            # mocked
            (True, False),
            # is_alive
            (None, True, False),
            # auto_start
            (True, False),
        )

        for name, mocked, is_alive, auto_start in test_specs:
            auto_gen_name = f'test_{len(Transceiver._instances)}'

            xcvr = self.xcvr_tester.build_xcvr(
                name=name, mocked=mocked, is_alive=is_alive, auto_start=auto_start
            )

            with self.subTest(
                name=name, mocked=mocked, is_alive=is_alive, auto_start=auto_start
            ):
                self.assertIsInstance(xcvr, Transceiver)

                # Checks dependent on `mocked`
                self.assertEqual(mocked, isinstance(xcvr, Mock))

                if not mocked and auto_start:
                    start_mock.assert_called_once()
                else:
                    start_mock.assert_not_called()

                # Checks independent of `mocked`
                if name is not None:
                    self.assertEqual(name, xcvr.name)
                else:
                    self.assertEqual(auto_gen_name, xcvr.name)

                if is_alive is not None:
                    self.assertEqual(is_alive, xcvr.is_alive())

            start_mock.reset_mock()


if __name__ == '__main__':
    main()
