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
        self, regex: bool, log_str: str, msgs_args: list[str | list[str]]
    ):
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
                        self.subTest(
                            level=level, is_list=isinstance(msgs, list)
                        )
                    )
                    stack.enter_context(
                        self.mixin_tester.assertInLogs(
                            __name__, level, msgs, regex=regex
                        )
                    )

                    log.log(level, log_str)

    def test_assert_in_logs_not_regex(self):
        self._run_logging_test(
            regex=False, log_str='test', msgs_args=['test', ['tes', 'est']]
        )

    def test_assert_in_logs_with_regex(self):
        self._run_logging_test(
            regex=True,
            log_str='test',
            msgs_args=['tes.', [r't\w{2}t', 'te.t']],
        )


if __name__ == '__main__':
    main()
