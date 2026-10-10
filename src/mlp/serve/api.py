"""Serving endpoints with single/batch validation and bounded Prometheus labels."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from mlp.data.schema import validate_frame
from mlp.monitor.runtime import runtime
from mlp.serve.manager import CACHE, ModelManager, manager
from mlp.serve.routing import CanaryGuard, Observation

CHALLENGER = ModelManager(cache=CACHE / "challenger")
GUARD = CanaryGuard(percent=float(os.getenv("MLP_CANARY_PERCENT", "0")))
SHADOW = os.getenv("MLP_SHADOW_MODE", "false").lower() == "true"

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
        if GUARD.percent or SHADOW:
            await asyncio.to_thread(CHALLENGER.reload, "challenger")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(manager.reload)
    if GUARD.percent or SHADOW:
        await asyncio.to_thread(CHALLENGER.reload, "challenger")
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
async def correlation_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    rid = request.headers.get("X-Request-ID", "")
    if not rid.isascii() or len(rid) > 64 or not rid:
        rid = uuid.uuid4().hex
    request.state.request_id = rid
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
async def predict(payload: PredictRequest, request: Request) -> dict[str, Any]:
    records = payload.records if isinstance(payload.records, list) else [payload.records]
    if not 1 <= len(records) <= 128:
        raise HTTPException(422, "Batch must contain 1 to 128 rows")
    champion = manager.snapshot()
    route = GUARD.route(request.state.request_id)
    challenger = CHALLENGER.snapshot()
    loaded = challenger if route == "challenger" and challenger is not None else champion
    if loaded is None:
        raise HTTPException(503, "No verified model is loaded")
    try:
        frame = validate_frame(pd.DataFrame.from_records(records), training=False)
    except (ValueError, TypeError) as exc:
        REQUESTS.labels(route, "invalid").inc()
        raise HTTPException(status_code=422, detail="Invalid input schema") from exc
    except Exception as exc:
        REQUESTS.labels("champion", "invalid").inc()
        raise HTTPException(status_code=422, detail="Input validation failed") from exc
    try:
        from time import perf_counter

        started = perf_counter()
        with LATENCY.labels(route).time():
            scores = await asyncio.wait_for(
                asyncio.to_thread(lambda: loaded.model.predict_proba(frame)[:, 1]),
                timeout=float(os.getenv("MLP_PREDICTION_TIMEOUT", "5")),
            )
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Model produced invalid scores")
    except (ValueError, TimeoutError, RuntimeError) as exc:
        REQUESTS.labels(route, "error").inc()
        if route == "challenger" and GUARD.observe(Observation(True, 0, 0)):
            LOG.error("canary_auto_rollback")
        LOG.exception("prediction_failed")
        raise HTTPException(status_code=503, detail="Model prediction failed") from exc
    runtime.observe(frame, scores)
    REQUESTS.labels(route, "ok").inc()
    if route == "challenger":
        elapsed_ms = (perf_counter() - started) * 1000
        if GUARD.observe(Observation(False, elapsed_ms, 0)):
            LOG.error("canary_auto_rollback")
    elif SHADOW and challenger is not None:
        # Deliberately never return or persist the shadow result.
        async def score_shadow() -> None:
            try:
                await asyncio.to_thread(challenger.model.predict_proba, frame)
                REQUESTS.labels("shadow", "ok").inc()
            except (ValueError, RuntimeError):
                REQUESTS.labels("shadow", "error").inc()
                LOG.warning("shadow_predict_failed")

        asyncio.create_task(score_shadow())
    prediction_ids = [uuid.uuid4().hex for _ in scores]
    if os.getenv("MLP_ENABLE_DELAYED_FEEDBACK", "false").lower() == "true":
        for prediction_id, score in zip(prediction_ids, scores, strict=True):
            runtime.feedback.add_prediction(prediction_id, float(score))
    return {
        "prediction_id": prediction_ids[0],
        "prediction_ids": prediction_ids,
        "version": loaded.version,
        "route": route,
        "scores": [float(x) for x in scores],
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class LabelFeedback(BaseModel):
    prediction_id: str
    target: int


@app.post("/v1/feedback")
def delayed_feedback(payload: LabelFeedback) -> dict[str, bool]:
    if os.getenv("MLP_ENABLE_DELAYED_FEEDBACK", "false").lower() != "true":
        raise HTTPException(403, "Delayed feedback is disabled")
    if payload.target not in (0, 1):
        raise HTTPException(422, "Binary label required")
    if not runtime.feedback.add_label(payload.prediction_id, payload.target):
        raise HTTPException(404, "Prediction ID expired or unknown")
    return {"accepted": True}


@app.delete("/v1/predictions/{prediction_id}")
def erase_prediction(prediction_id: str) -> dict[str, bool]:
    runtime.feedback.erase(prediction_id)
    return {"erased": True}


@app.get("/v1/performance")
def delayed_performance() -> dict[str, str | int | float]:
    return runtime.feedback.performance()


@app.get("/v1/drift")
def live_drift() -> dict[str, dict]:
    return runtime.report()
