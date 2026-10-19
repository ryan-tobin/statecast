import asyncio

import httpx
import pytest
import respx

from statecast.config import IngestSettings
from statecast.ingest.http_client import (
    RetryableStatusError,
    get_bytes,
    open_client,
    run_bounded,
)

URL = "https://example.test/data"


@respx.mock
async def test_get_bytes_returns_body(ingest_settings: IngestSettings) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=b"ok"))

    async with open_client(ingest_settings) as client:
        body = await get_bytes(client, URL, {"a": "1"}, max_retries=2)

    assert body == b"ok"
    assert route.calls.last.request.url.params["a"] == "1"
    assert route.calls.last.request.headers["User-Agent"] == "statecast-tests"


@respx.mock
async def test_get_bytes_retries_server_error(ingest_settings: IngestSettings) -> None:
    route = respx.get(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, content=b"ok")]
    )

    async with open_client(ingest_settings) as client:
        body = await get_bytes(client, URL, {}, max_retries=3, backoff_seconds=0.0)

    assert body == b"ok"
    assert route.call_count == 2


@respx.mock
async def test_get_bytes_does_not_retry_not_found(
    ingest_settings: IngestSettings,
) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(404))

    async with open_client(ingest_settings) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await get_bytes(client, URL, {}, max_retries=3, backoff_seconds=0.0)

    assert route.call_count == 1


@respx.mock
async def test_get_bytes_raises_after_exhausting_retries(
    ingest_settings: IngestSettings,
) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(429))

    async with open_client(ingest_settings) as client:
        with pytest.raises(RetryableStatusError, match="429"):
            await get_bytes(client, URL, {}, max_retries=2, backoff_seconds=0.0)

    assert route.call_count == 2


async def test_run_bounded_returns_every_result_within_limit() -> None:
    in_flight = 0
    peak_in_flight = 0

    async def task() -> int:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return 1

    results = await run_bounded([task] * 6, max_concurrency=2, description="test")

    assert results == [1] * 6
    assert peak_in_flight == 2
