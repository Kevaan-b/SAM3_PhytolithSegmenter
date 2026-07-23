from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .engine import SamEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPU_BUDGET_GIB = float(os.environ.get("SAMOTATOR_GPU_CACHE_GIB", "16"))
engine = SamEngine(PROJECT_ROOT, GPU_BUDGET_GIB)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initializer = threading.Thread(target=engine.initialize, daemon=True)
    initializer.start()
    yield
    engine.shutdown()


app = FastAPI(title="Samotator H100 service", lifespan=lifespan)


class PrepareRequest(BaseModel):
    imageId: str
    imageRevision: int


class Point(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    label: int = Field(ge=0, le=1)


class SegmentRequest(BaseModel):
    imageId: str
    imageRevision: int
    stateRevision: int
    points: list[Point] = Field(min_length=1)


def require_ready() -> None:
    if engine.error:
        raise HTTPException(status_code=503, detail=engine.error)
    if not engine.ready:
        raise HTTPException(status_code=503, detail="SAM3 is still loading on the H100.")


@app.get("/api/status")
def status():
    return engine.status()


@app.get("/api/data-tree")
def data_tree(refresh: bool = False):
    return engine.refresh_manifest() if refresh else engine.manifest()


@app.post("/api/images/prepare")
async def prepare(request: PrepareRequest):
    require_ready()
    try:
        result = await engine.prepare(request.imageId)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {**result, "imageRevision": request.imageRevision}


@app.post("/api/images/segment")
async def segment(request: SegmentRequest):
    require_ready()
    try:
        payload, metadata = await engine.segment(
            request.imageId, [point.model_dump() for point in request.points]
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    headers = {
        "X-Image-Revision": str(request.imageRevision),
        "X-State-Revision": str(request.stateRevision),
        "X-Mask-Width": str(metadata["width"]),
        "X-Mask-Height": str(metadata["height"]),
        "X-Decode-Ms": f'{metadata["decodeMs"]:.3f}',
        "X-Cache-State": metadata["cacheState"],
    }
    return Response(payload, media_type="application/octet-stream", headers=headers)
