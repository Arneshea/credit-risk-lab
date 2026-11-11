"""Serve the custom Soft Computing Lab UI and its local prediction API.

Run with: python web_app.py
Then open: http://127.0.0.1:8765
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from predict import explain_customer
from src.data import RAW_COLS
from src.lab import run_comparison
from src.fuzzy import FuzzyRuleBase


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "frontend"
PORT = 8765
_taiwan_bundle = None
_taiwan_model = None


def _legacy_model():
    global _taiwan_bundle, _taiwan_model
    if _taiwan_model is None:
        path = ROOT / "results" / "model_taiwan.json"
        _taiwan_bundle = json.loads(path.read_text(encoding="utf-8"))
        _taiwan_model = FuzzyRuleBase.from_dict(_taiwan_bundle["model"])
    return _taiwan_model, _taiwan_bundle


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, value):
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path in ("/styles.css", "/app.js"):
            mime = "text/css; charset=utf-8" if path.endswith(".css") else "text/javascript; charset=utf-8"
            self._send(200, (STATIC / path.lstrip("/")).read_bytes(), mime)
        elif path == "/api/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                return self._json(400, {"error": "Request body must be between 1 byte and 1 MB."})
            body = json.loads(self.rfile.read(length))
            if path == "/api/benchmark":
                return self._json(200, run_comparison(body.get("dataset", "taiwan")))
            if path == "/api/legacy-predict":
                missing = [column for column in RAW_COLS if column not in body]
                if missing:
                    return self._json(400, {"error": "Missing required fields: " + ", ".join(missing)})
                raw = pd.DataFrame([{column: body[column] for column in RAW_COLS}])
                model, bundle = _legacy_model()
                explanation = explain_customer(raw, model, bundle)
                explanation["requested_amount"] = float(body.get("requested_amount", 0))
                explanation["customer_ref"] = str(body.get("customer_ref", ""))[:100]
                return self._json(200, explanation)
            return self._json(404, {"error": "Not found"})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:
            return self._json(500, {"error": f"Unable to complete request: {exc}"})

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Soft Computing Lab UI running at http://127.0.0.1:{PORT}")
    print("The existing Streamlit app remains available with: streamlit run app.py")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Soft Computing Lab UI.")
    finally:
        server.server_close()
