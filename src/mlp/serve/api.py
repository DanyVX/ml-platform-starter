"""Serving endpoints with single/batch validation and bounded Prometheus labels."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from mlp.data.schema import validate_frame
from mlp.serve.manager import manager

LOG = logging.getLogger("mlp.serve")
REQUESTS = Counter("mlp_prediction_requests_total", "Prediction requests", ["route", "status"])
LATENCY = Histogram("mlp_prediction_latency_seconds", "Inference latency", ["route"])


class PredictRequest(BaseModel):
    records: dict[str, Any] | list[dict[str, Any]]


async def _poll_alias() -> None:
    interval = max(5, int(os.getenv("MLP_RELOAD_INTERVAL", "30")))
    while True:
        await asyncio.sleep(interval)
        await asyncio.to_thread(manager.reload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(manager.reload)
    task = asyncio.create_task(_poll_alias())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="ML Platform Starter", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def correlation_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", "")
    if not rid.isascii() or len(rid) > 64 or not rid:
        rid = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
def ready() -> dict[str, str]:
    loaded = manager.snapshot()
    if loaded is None:
        raise HTTPException(status_code=503, detail="No verified model available")
    return {"status": loaded.status}


@app.get("/v1/model")
def model_info() -> dict[str, str]:
    loaded = manager.snapshot()
    if loaded is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {
        "alias": "champion",
        "version": loaded.version,
        "data_version": loaded.data_version,
        "training_commit": loaded.training_commit,
        "readiness": loaded.status,
    }


@app.post("/v1/predict")
async def predict(payload: PredictRequest) -> dict[str, Any]:
    records = payload.records if isinstance(payload.records, list) else [payload.records]
    if not 1 <= len(records) <= 128:
        raise HTTPException(422, "Batch must contain 1 to 128 rows")
    loaded = manager.snapshot()
    if loaded is None:
        raise HTTPException(503, "No verified model is loaded")
    try:
        frame = validate_frame(pd.DataFrame.from_records(records), training=False)
    except (ValueError, TypeError) as exc:
        REQUESTS.labels("champion", "invalid").inc()
        raise HTTPException(status_code=422, detail="Invalid input schema") from exc
    except Exception as exc:
        REQUESTS.labels("champion", "invalid").inc()
        raise HTTPException(status_code=422, detail="Input validation failed") from exc
    try:
        with LATENCY.labels("champion").time():
            scores = await asyncio.wait_for(
                asyncio.to_thread(lambda: loaded.model.predict_proba(frame)[:, 1]),
                timeout=float(os.getenv("MLP_PREDICTION_TIMEOUT", "5")),
            )
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Model produced invalid scores")
    except (ValueError, asyncio.TimeoutError, RuntimeError) as exc:
        REQUESTS.labels("champion", "error").inc()
        LOG.exception("prediction_failed")
        raise HTTPException(status_code=503, detail="Model prediction failed") from exc
    REQUESTS.labels("champion", "ok").inc()
    return {
        "prediction_id": uuid.uuid4().hex,
        "version": loaded.version,
        "scores": [float(x) for x in scores],
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
