#!/usr/bin/env python
import logging
from unittest import TestCase, main
from contextlib import ExitStack

from pynet.testing import LogTestingMixin

log = logging.getLogger(__name__)


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


if __name__ == '__main__':
    main()
