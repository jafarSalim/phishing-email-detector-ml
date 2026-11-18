"""HTTP API: POST a raw e-mail, get a phishing score and the reasons.

    uvicorn api.app:app --port 8000
    curl -s --data-binary @message.eml http://localhost:8000/scan | jq

Response:
    {"score": 0.97, "verdict": "phishing", "score_features": 0.99, "score_text": 0.95,
     "reasons": [{"feature": "anchor_href_mismatch", "description": "...", "value": 2, "contribution": 1.8}, ...],
     "words": [{"term": "verify your", "contribution": 0.9}, ...], "model": "phishguard 1.0.0"}
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from phishguard import __version__, features, parse  # noqa: E402
from phishguard.model import Detector  # noqa: E402

MODEL = os.environ.get("PHISHGUARD_MODEL", os.path.join(os.path.dirname(__file__), "..", "models", "phishguard.joblib"))
THRESHOLD = float(os.environ.get("PHISHGUARD_THRESHOLD", "0.5"))
MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="phishguard", version=__version__)
_det = None


def detector() -> Detector:
    global _det
    if _det is None:
        _det = Detector.load(MODEL)
    return _det


def to_row(raw: bytes) -> pd.DataFrame:
    m = parse.parse(raw)
    return pd.DataFrame([dict(subject=m.subject or "", text=(m.text or "")[:20000], **features.extract(m))])


@app.get("/health")
def health():
    return {"ok": True, "model": f"phishguard {__version__}", "threshold": THRESHOLD}


@app.post("/scan")
async def scan(request: Request):
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "send the raw message (RFC 822 / .eml) as the request body")
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "message larger than 10 MB")
    out = detector().explain(to_row(raw))
    out["verdict"] = "phishing" if out["score"] >= THRESHOLD else "legitimate"
    out["model"] = f"phishguard {__version__}"
    return out
