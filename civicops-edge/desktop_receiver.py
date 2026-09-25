"""
Tiny telemetry receiver for your PC (Windows/Linux/Mac) - standard library only.

  python desktop_receiver.py            # listens on 0.0.0.0:8000

Then on the Pi:
  TELEMETRY_URL=http://<PC-IP>:8000/events python main.py

  POST /events  -> prints the event, appends to received_events.jsonl
  GET  /        -> last 50 events as JSON (open in a browser)
  GET  /health  -> "ok"

Windows will ask to allow Python through the firewall the first time - allow it
on Private networks.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = Path(__file__).with_name("received_events.jsonl")
RECENT: deque = deque(maxlen=50)
COLORS = {"LOW": "\033[32m", "MEDIUM": "\033[33m", "CRITICAL": "\033[31m"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, ctype: str = "application/json") -> None:
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/events":
            return self._send(404, '{"error":"not found"}')
        try:
            evt = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self._send(400, '{"error":"bad json"}')
        RECENT.appendleft(evt)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt) + "\n")
        sev = evt.get("severity_level", "?")
        g, v, s, k = evt.get("gps", {}), evt.get("vision", {}), evt.get("surface", {}), evt.get("kinetics", {})
        print(f"{datetime.now():%H:%M:%S} {COLORS.get(sev, '')}{sev:<8}\033[0m "
              f"{evt.get('node_id')}  conf={v.get('confidence')}  depth={s.get('depth_cm')}cm  "
              f"z={k.get('z_impact_g')}g  @ {g.get('latitude')},{g.get('longitude')}"
              f"{' (mock gps)' if g.get('is_mock') else ''}")
        self._send(201, '{"ok":true}')

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send(200, "ok", "text/plain")
        self._send(200, json.dumps(list(RECENT), indent=2))

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"CivicOps receiver on http://{a.host}:{a.port}/events  (log: {LOG})")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
