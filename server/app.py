from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .engine import SamEngine
from .annotations import AnnotationStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPU_BUDGET_GIB = float(os.environ.get("SAMOTATOR_GPU_CACHE_GIB", "16"))
engine = SamEngine(PROJECT_ROOT, GPU_BUDGET_GIB)
annotations = AnnotationStore(PROJECT_ROOT / "data", engine.get_record, PROJECT_ROOT / ".samotator-cache")


@asynccontextmanager
async def lifespan(_: FastAPI):
    initializer = threading.Thread(target=engine.initialize, daemon=True)
    indexer = threading.Thread(target=annotations.initialize, daemon=True, name="annotation-index")
    initializer.start()
    indexer.start()
    yield
    annotations.shutdown()
    engine.shutdown()


app = FastAPI(title="Samotator H100 service", lifespan=lifespan)


class CachePrioritizeRequest(BaseModel):
    folderPath: str
    imageId: str | None = None


class CacheInteractionRequest(BaseModel):
    active: bool


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


class CategoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    supercategory: str = Field(default="phytolith", min_length=1, max_length=80)
    color: str | None = None


class CategoryUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    supercategory: str | None = Field(default=None, min_length=1, max_length=80)
    color: str | None = None
    active: bool | None = None


class AnnotationLayer(BaseModel):
    layerId: str = Field(min_length=1, max_length=128)
    categoryId: int = Field(ge=1)
    rawMask: str
    effectiveMask: str


class AnnotationSave(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    latestMaskLayerId: str | None = None
    preventOverlap: bool = False
    layers: list[AnnotationLayer]


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


@app.get("/api/classes")
def classes():
    try:
        return annotations.categories()
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/statistics")
def annotation_statistics():
    try:
        return annotations.statistics()
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/statistics/classes/{category_id}/previews")
def annotation_statistics_previews(category_id: int, limit: int = 8):
    try:
        return {"previews": annotations.statistics_previews(category_id, limit)}
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Class not found.") from error


@app.get("/api/statistics/previews/{image_id}")
def annotation_preview(image_id: str, category_id: int | None = None, v: str | None = None):
    try:
        return Response(annotations.preview_image(image_id, category_id), media_type="image/webp",
            headers={"Cache-Control": "public, max-age=31536000, immutable" if v else "no-cache"})
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Image not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/classes")
def add_class(request: CategoryCreate):
    try:
        return annotations.add_category(request.name, request.supercategory, request.color)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.patch("/api/classes/{category_id}")
def update_class(category_id: int, request: CategoryUpdate):
    try:
        return annotations.update_category(category_id, request.model_dump(exclude_none=True))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/classes/{category_id}")
def archive_class(category_id: int):
    try:
        return annotations.archive_category(category_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/images/{image_id}/annotations")
def image_annotations(image_id: str):
    try:
        return annotations.load_image(image_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.put("/api/images/{image_id}/annotations")
def save_image_annotations(image_id: str, request: AnnotationSave):
    try:
        return annotations.save_image(
            image_id, request.width, request.height,
            [layer.model_dump() for layer in request.layers],
            request.latestMaskLayerId, request.preventOverlap,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/cache/prioritize")
def prioritize_cache(request: CachePrioritizeRequest):
    require_ready()
    try:
        return engine.prioritize_folder(request.folderPath, request.imageId)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/cache/interaction")
def cache_interaction(request: CacheInteractionRequest):
    engine.set_interactive(request.active)
    annotations.set_interactive(request.active)
    return {"active": request.active}


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
