"""Shared async HTTP client, retrying GET and bounded-concurrency runner."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

from statecast.config import IngestSettings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class RetryableStatusError(Exception):
    """Raised for an HTTP status that is worth retrying (rate limit or 5xx)."""


@asynccontextmanager
async def open_client(settings: IngestSettings) -> AsyncIterator[httpx.AsyncClient]:
    """Open an async HTTP client configured from ingest settings.

    Args:
        settings: Ingest settings supplying the timeout and User-Agent.

    Yields:
        An open ``httpx.AsyncClient`` that is closed on exit.
    """
    async with httpx.AsyncClient(
        timeout=settings.timeout_seconds,
        headers={"User-Agent": settings.user_agent},
        follow_redirects=True,
    ) as client:
        yield client


async def get_bytes(
    client: httpx.AsyncClient,
    url: str,
    params: Mapping[str, str],
    max_retries: int,
    backoff_seconds: float = 1.0,
) -> bytes:
    """GET a URL and return the body, retrying transient failures.

    Retries network errors, HTTP 429 and HTTP 5xx with exponential backoff and
    jitter. Each retry is logged at WARNING level.

    Args:
        client: Open HTTP client.
        url: URL to request.
        params: Query parameters.
        max_retries: Total attempts before the last error is raised.
        backoff_seconds: Initial backoff; doubles on each retry up to 30 s.

    Returns:
        The response body.

    Raises:
        RetryableStatusError: If every attempt returned a retryable status.
        httpx.HTTPStatusError: For a non-retryable error status such as 404.
        httpx.TransportError: If every attempt failed at the network level.
    """
    retrying = AsyncRetrying(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(multiplier=backoff_seconds, max=30.0),
        retry=retry_if_exception_type((httpx.TransportError, RetryableStatusError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            response = await client.get(url, params=params)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise RetryableStatusError(
                    f"HTTP {response.status_code} from {response.request.url.host}"
                )
            response.raise_for_status()
            return response.content
    raise RuntimeError("Retry loop ended without a response")


async def run_bounded[T](
    tasks: Sequence[Callable[[], Awaitable[T]]],
    max_concurrency: int,
    description: str,
) -> list[T]:
    """Run task factories concurrently, at most ``max_concurrency`` at a time.

    Results come back in completion order, with a progress bar.

    Args:
        tasks: Zero-argument callables that each return an awaitable.
        max_concurrency: Maximum tasks in flight.
        description: Progress-bar label.

    Returns:
        Each task's result, in completion order.

    Example:
        >>> async def fetch_one() -> int:
        ...     return 1
        >>> asyncio.run(run_bounded([fetch_one, fetch_one], 2, "demo"))
        [1, 1]
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def guarded(task: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await task()

    results: list[T] = []
    with tqdm(total=len(tasks), desc=description, unit="request") as progress:
        for future in asyncio.as_completed([guarded(task) for task in tasks]):
            results.append(await future)
            progress.update()
    return results
