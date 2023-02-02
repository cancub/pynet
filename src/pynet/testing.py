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

        try:
            with self.assertLogs(logger, level) as cm:
                yield
        except AssertionError:
            # This might not be bad if we're expecting nothing to be logged.
            if in_logs:
                # Nope, it's bad.
                raise
            else:
                # We can call it a day because nothing has been logged at all at this
                # level.
                return

        # _Something_ has been logged at this level, so we need to check if it is or is
        # not what we're looking for.
        all_logs = '\n'.join(cm.output)

        not_str = '' if in_logs else 'Not'

        # Unfortunately, the order of the arguments to assert[Not]In and assert[Not]Regex
        # is reversed, so we need to explicitly check which one we're using.
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
