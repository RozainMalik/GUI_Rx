"""BatchClient: the Files and Batches endpoints over a synchronous httpx client. Same retry policy as OpenAIGenerator - 429 backs off, 5xx and connection errors retry three
times, any other 4xx fails at once with the body - and the same TransportError, so a step that
cannot reach the API exits like a sync run does. No SDK: tests script an httpx transport."""
from __future__ import annotations

import time

import httpx

from annotator.calling import TransportError

from .client import BASE_URL, RATE_LIMIT_ATTEMPTS, SERVER_ERROR_ATTEMPTS, retry_after

COMPLETION_WINDOW = "24h"
ENDPOINT = "/v1/responses"
MIN_UPLOAD_BYTES_PER_S = 100_000    # the write timeout of an upload allows for an uplink this slow (measured: ~800 KB/s)


class BatchClient:
    def __init__(self, api_key, *, base_url=BASE_URL, transport=None, sleep=time.sleep, timeout=120.0):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        self.sleep, self.timeout = sleep, timeout
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout, connect=30.0),
                                   headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    def upload(self, name: str, data: bytes, purpose: str, expires_seconds: int | None = None) -> dict:
        """httpx sends a bytes body as one socket write, so the write timeout must cover the whole
        payload at a slow uplink - a 180 MB part is minutes, not the default 120 s."""
        form = {"purpose": purpose}
        if expires_seconds:
            form["expires_after[anchor]"], form["expires_after[seconds]"] = "created_at", str(int(expires_seconds))
        write = max(self.timeout, 60.0 + len(data) / MIN_UPLOAD_BYTES_PER_S)
        timeout = httpx.Timeout(self.timeout, connect=30.0, write=write)
        return self._request("POST", "/files", files={"file": (name, data, "application/octet-stream")}, data=form, timeout=timeout).json()

    def create_batch(self, input_file_id: str, metadata: dict) -> dict:
        body = {"input_file_id": input_file_id, "endpoint": ENDPOINT, "completion_window": COMPLETION_WINDOW, "metadata": metadata}
        return self._request("POST", "/batches", json=body).json()

    def get_batch(self, batch_id: str) -> dict:
        return self._request("GET", f"/batches/{batch_id}").json()

    def cancel_batch(self, batch_id: str) -> dict:
        return self._request("POST", f"/batches/{batch_id}/cancel").json()

    def download(self, file_id: str) -> bytes:
        return self._request("GET", f"/files/{file_id}/content").content

    def _request(self, method, url, **kw) -> httpx.Response:
        rate_limits = server_errors = 0
        while True:
            try:
                r = self.client.request(method, url, **kw)
            except httpx.HTTPError as e:                          # timeout, connection refused, ...
                server_errors += 1
                if server_errors >= SERVER_ERROR_ATTEMPTS:
                    raise TransportError(f"OpenAI unreachable after {server_errors} attempts: {e!r}") from e
                self.sleep(float(2 ** (server_errors - 1)))
                continue
            if r.status_code == 429:
                rate_limits += 1
                if rate_limits >= RATE_LIMIT_ATTEMPTS:
                    raise TransportError(f"OpenAI 429 after {rate_limits} attempts: {r.text[:500]}")
                self.sleep(retry_after(r, rate_limits))
                continue
            if r.status_code >= 500:
                server_errors += 1
                if server_errors >= SERVER_ERROR_ATTEMPTS:
                    raise TransportError(f"OpenAI {r.status_code} after {server_errors} attempts: {r.text[:500]}")
                self.sleep(float(2 ** (server_errors - 1)))
                continue
            if r.status_code >= 400:
                raise TransportError(f"OpenAI {r.status_code} {method} {url}: {r.text[:500]}")
            return r

    def close(self) -> None:
        self.client.close()
