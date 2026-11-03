"""HTTP API. It serves only a promoted, hash-verified model and validates every request."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from pagrus_age.predict import load_bundle, predict_frame


class Fish(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_length_mm: float = Field(gt=0, le=2000)
    month: int | None = Field(default=None, ge=1, le=12)
    gear: Literal["handline", "otter_trawl", "pair_trawl"] | None = None
    depth_min_m: float | None = Field(default=None, ge=0)
    depth_max_m: float | None = Field(default=None, ge=0)
    total_weight_g: float | None = Field(default=None, gt=0)


class Batch(BaseModel):
    fish: list[Fish] = Field(min_length=1, max_length=1000)


def create_app(models_dir: str | None = None, run_id: str | None = None) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        directory = models_dir or os.environ.get("PAGRUS_MODELS_DIR", "models")
        state["bundle"] = load_bundle(directory, run_id or os.environ.get("PAGRUS_RUN_ID"))
        yield

    app = FastAPI(title="Pagrus age", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        bundle = state.get("bundle")
        if bundle is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok", "model": bundle["model_name"], "run_id": bundle["run_id"]}

    @app.post("/predict")
    def predict(batch: Batch) -> dict[str, Any]:
        bundle = state["bundle"]
        frame = pd.DataFrame([fish.model_dump() for fish in batch.fish])
        records = predict_frame(bundle, frame).to_dict(orient="records")
        return {"run_id": bundle["run_id"], "predictions": records}

    return app
