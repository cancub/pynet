from contextlib import contextmanager
from typing import Sequence
from functools import partialmethod


class LogTestingMixin:
    @contextmanager
    def _logChecker(
        self,
        in_logs: bool,
        logger: str,
        level: str,
        msgs: str | Sequence[str],
        regex: bool = False,
    ):

        if isinstance(msgs, str):
            msgs = [msgs]

        with self.assertLogs(logger, level) as cm:
            yield

        all_logs = '\n'.join(cm.output)

        not_str = '' if in_logs else 'Not'
        if regex:
            assertion = getattr(self, f'assert{not_str}Regex')
            for regex in msgs:
                assertion(all_logs, regex)
        else:
            assertion = getattr(self, f'assert{not_str}In')
            for substr in msgs:
                assertion(substr, all_logs)

    assertInLogs = partialmethod(_logChecker, True)

    assertNotInLogs = partialmethod(_logChecker, False)
