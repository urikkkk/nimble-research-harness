"""Retry utilities with exponential backoff."""

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .errors import NimbleApiError


def _is_retryable_nimble_error(exc: BaseException) -> bool:
    if isinstance(exc, NimbleApiError):
        return exc.status_code in (429, 500, 502, 503, 504)
    return False


nimble_retry = retry(
    retry=retry_if_exception(_is_retryable_nimble_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=8, jitter=2),
    reraise=True,
)
