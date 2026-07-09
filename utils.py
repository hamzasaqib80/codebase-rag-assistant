"""
utils.py — Shared utilities: structured logging and exponential-backoff retry decorator.
"""

import logging
import sys
import time
import functools
from typing import Tuple, Type


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return the root logger with a structured format that includes
    timestamps, log level, source filename, and line number.
    Idempotent — calling multiple times will not add duplicate handlers.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured; just set the level and return.
        root.setLevel(level)
        return root

    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return root


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
):
    """
    Decorator that retries the wrapped function on failure with exponential backoff.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts after the first failure.
    delay : float
        Initial wait time in seconds before the first retry.
    backoff : float
        Multiplicative factor applied to `delay` after each subsequent failure.
    exceptions : tuple of exception types
        Only these exception classes (and their subclasses) will trigger a retry.
        All other exceptions propagate immediately.

    Example
    -------
    >>> @retry_on_failure(max_retries=5, delay=1, backoff=2)
    ... def call_external_api():
    ...     ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            current_delay = delay
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_retries:
                        logger.error(
                            "[%s] All %d retry attempts exhausted. Last error: %s",
                            func.__qualname__,
                            max_retries,
                            exc,
                        )
                        raise
                    logger.warning(
                        "[%s] Attempt %d/%d failed: %s — retrying in %.1fs …",
                        func.__qualname__,
                        attempt + 1,
                        max_retries,
                        exc,
                        current_delay,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            # Unreachable, but keeps static analysers happy.
            raise RuntimeError("retry_on_failure: unexpected exit") from last_exception

        return wrapper

    return decorator
