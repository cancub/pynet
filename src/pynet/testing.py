from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Sequence
from functools import partialmethod
from unittest.mock import Mock

from .physical.base import Medium, Transceiver

__all__ = ['LogTestingMixin', 'ProcessBuilderMixin']


class LogTestingMixin(ABC):
    """A mixin for unittest.TestCase that provides methods for asserting logs.
    TODO:
    - fuse both log checks into one method
        - have one param to a list of logs that _must_ be seen
        - have another param to a list of logs that _must not_ be seen
        - currently the issue is that if we want nested log checks, we need to use
            assertNotInLogs on the outside and assertInLogs on the inside, which is
            unintuitive
    """

    @property
    @abstractmethod
    def log_target(self) -> str:
        """The target of the logs we're checking. This is the name of the logger that we
        pass to assert[Not]InTargetLogs. Override as a class attribute."""
        raise NotImplementedError

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

    @contextmanager
    def assertInTargetLogs(self, level, msgs, *args, **kwargs):
        with self.assertInLogs(self.log_target, level, msgs, *args, **kwargs):
            yield

    @contextmanager
    def assertNotInTargetLogs(self, level, msgs, *args, **kwargs):
        with self.assertNotInLogs(self.log_target, level, msgs, *args, **kwargs):
            yield


class ProcessBuilderMixin:
    medium_cls: type[Medium] = None
    xcvr_cls: type[Transceiver] = None

    def build_medium(
        self,
        name=None,
        mocked=False,
        auto_start=False,
        is_alive=None,
    ):
        # Make sure that we have a unique name for each medium
        name = name or f'test_{len(Medium._instances)}'

        if mocked:
            medium = Mock(spec=self.medium_cls)
            medium.name = name
            medium._connection_queue = Mock()
        else:
            medium = self.medium_cls(name=name, auto_start=auto_start)

        if is_alive is not None:
            medium.is_alive = Mock(return_value=is_alive)

        return medium

    def build_xcvr(
        self,
        name=None,
        base_baud=2e6,
        mocked=False,
        location=None,
        auto_start=False,
        is_alive=None,
        mock_medium=False,
    ):
        # Make sure that we have a unique name for each transceiver
        name = name or f'test_{len(Transceiver._instances)}'

        if mocked:
            xcvr = Mock(spec=self.xcvr_cls)
            xcvr.name = name
        else:
            xcvr = self.xcvr_cls(name=name, base_baud=base_baud, auto_start=auto_start)

        if location is not None:
            xcvr.location = location

        if is_alive is not None:
            xcvr.is_alive = Mock(return_value=is_alive)
        if mock_medium:
            xcvr._medium = self.build_medium(mocked=True)

        return xcvr
