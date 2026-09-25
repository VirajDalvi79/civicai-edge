"""
Telemetry dispatcher: a background thread drains a queue of payloads.

  1. Always appends to output/events.jsonl (one JSON object per line).
  2. If TELEMETRY_URL is set, POSTs the JSON with retries + backoff.
     Failed posts go to output/unsent.jsonl and are retried on the next
     successful connection, so nothing is lost when the Pi is offline.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger("dispatch")

_SENTINEL = object()


class TelemetryDispatcher:
    def __init__(self, jsonl_path: Path | None = None, url: str | None = None) -> None:
        self.jsonl_path = Path(jsonl_path or config.TELEMETRY_JSONL)
        self.unsent_path = self.jsonl_path.with_name("unsent.jsonl")
        self.url = config.TELEMETRY_URL if url is None else url
        self.q: "queue.Queue" = queue.Queue(maxsize=config.TELEMETRY_QUEUE_SIZE)
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self.written = 0
        self.posted = 0
        self.failed = 0
        self.dropped = 0

    def start(self) -> "TelemetryDispatcher":
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.url:
            import requests
            self._session = requests.Session()
            headers = {"Content-Type": "application/json"}
            if config.TELEMETRY_API_KEY:
                headers["Authorization"] = f"Bearer {config.TELEMETRY_API_KEY}"
            self._session.headers.update(headers)
            log.info("Telemetry -> %s and %s", self.jsonl_path, self.url)
        else:
            log.info("Telemetry -> %s (no TELEMETRY_URL set)", self.jsonl_path)
        self._thread = threading.Thread(target=self._loop, name="dispatch", daemon=True)
        self._thread.start()
        return self

    def submit(self, payload: dict) -> bool:
        """Non-blocking enqueue. Returns False if the queue is full."""
        try:
            self.q.put_nowait(payload)
            return True
        except queue.Full:
            self.dropped += 1
            log.error("Telemetry queue full - event dropped")
            return False

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> bool:
        delay = 0.5
        for attempt in range(1, config.TELEMETRY_MAX_RETRIES + 1):
            try:
                r = self._session.post(self.url, data=json.dumps(payload),
                                       timeout=config.TELEMETRY_TIMEOUT_S)
                if 200 <= r.status_code < 300:
                    return True
                log.warning("POST %s -> HTTP %s (attempt %d)", self.url, r.status_code, attempt)
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    return False            # client error: retrying won't help
            except Exception as e:
                log.warning("POST failed (attempt %d): %s", attempt, e)
            time.sleep(delay)
            delay *= 2
        return False

    def _flush_unsent(self) -> None:
        if not self.unsent_path.exists():
            return
        lines = self.unsent_path.read_text().splitlines()
        remaining = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            if self._post(json.loads(line)):
                self.posted += 1
            else:
                remaining = lines[i:]
                break
        if remaining:
            self.unsent_path.write_text("\n".join(remaining) + "\n")
        else:
            self.unsent_path.unlink(missing_ok=True)
            log.info("Flushed backlog of unsent events")

    def _handle(self, payload: dict) -> None:
        line = json.dumps(payload, separators=(",", ":"))
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self.written += 1
        if self._session is not None:
            if self._post(payload):
                self.posted += 1
                self._flush_unsent()
            else:
                self.failed += 1
                with self.unsent_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def _loop(self) -> None:
        while True:
            item = self.q.get()
            if item is _SENTINEL:
                break
            try:
                self._handle(item)
            except Exception as e:
                log.exception("dispatch error: %s", e)

    def stop(self, timeout: float = 5.0) -> None:
        """Flush remaining events, then stop."""
        if self._thread and self._thread.is_alive():
            self.q.put(_SENTINEL)
            self._thread.join(timeout=timeout)
        if self._session is not None:
            self._session.close()
        log.info("Dispatcher stopped (written=%d posted=%d failed=%d dropped=%d)",
                 self.written, self.posted, self.failed, self.dropped)
