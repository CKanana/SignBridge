"""api/main.py -- FastAPI application for the standalone SignBridge service.

Endpoints:
  GET  /health
  POST /v1/animations             {"text": "I need help"}  -> video/mp4
  POST /v1/animations/structured  {"text": "I need help"}  -> pose JSON

The POST endpoints require the ``X-API-Key`` header (or
``Authorization: Bearer <key>``) matching ``SIGNBRIDGE_API_KEY``.

Rendering is CPU-bound and the engine is single-process: requests are
serialised behind one lock so a single Uvicorn worker stays responsive for
``/health``. Generated videos are deleted after the response is sent.
"""

from __future__ import annotations

import hmac
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

# Make the repository root importable so ``api`` and the original engine (a
# bare ``import animation_lib`` inside main.py) resolve regardless of how
# Uvicorn was invoked (console script vs ``python -m uvicorn``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import Depends, FastAPI, Header, HTTPException, Request  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, field_validator  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

from api.animation_service import (  # noqa: E402
    ConfigError,
    EncodingError,
    engine_ready,
    encoder_ready,
    generate_animations,
    generate_pose_sequence,
    get_config,
    remove_output,
    sweep_outputs,
    validate_text,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "info").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )


_setup_logging()
logger = logging.getLogger("signbridge-service.app")

API_KEY = (os.environ.get("SIGNBRIDGE_API_KEY") or "").strip()

# One render at a time: the engine is CPU-bound and uses module-level OpenCV
# state. Held by both endpoints so a single worker cannot interleave renders.
RENDER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Request-size limit (enforced on the ASGI receive channel, not just headers)
# ---------------------------------------------------------------------------
class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``MAX_REQUEST_SIZE`` bytes."""

    def __init__(self, app):
        self.app = app

    async def _reject(self, send) -> None:
        body = (
            b'{"status":"error","error":"Request body too large",'
            b'"detail":null}'
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        try:
            limit = get_config()["max_request_size"]
        except ConfigError:
            # Configuration errors are reported by the endpoints (/health).
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared = int(raw_length)
            except ValueError:
                return await self._reject(send)
            if declared > limit:
                return await self._reject(send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b"") or b"")
                if received > limit:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            logger.warning("Rejected oversized request body (> %d bytes)", limit)
            return await self._reject(send)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def require_api_key(
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    authorization: Annotated[Optional[str], Header()] = None,
) -> None:
    """Require the shared secret configured in ``SIGNBRIDGE_API_KEY``."""
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "API authentication is not configured: set SIGNBRIDGE_API_KEY "
                "on the service."
            ),
        )

    provided = (x_api_key or "").strip()
    if not provided and authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            provided = value.strip()

    if not provided or not hmac.compare_digest(provided, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        cfg = get_config()
        logger.info(
            "SignBridge service starting (root=%s, output=%s, api_key=%s)",
            cfg["signbridge_root"],
            cfg["output_dir"],
            "configured" if API_KEY else "MISSING",
        )
        engine_ready()
        if not encoder_ready():
            logger.error(
                "ffmpeg/ffprobe not available: video requests will fail instead "
                "of returning an unplayable file."
            )
        sweep_outputs()
    except ConfigError as exc:
        logger.error("Invalid configuration: %s", exc)
    yield
    logger.info("SignBridge service shutting down")


app = FastAPI(
    title="SignBridge Animation Service",
    version="2.0.0",
    description=(
        "Standalone HTTP API for the SignBridge text-to-ASL animation "
        "pipeline. Accepts English text, renders an H.264 MP4 with the "
        "original engine, and returns verified metadata."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(BodySizeLimitMiddleware)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class AnimRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _check_length(cls, v: str) -> str:
        try:
            max_length = get_config()["max_text_length"]
        except ConfigError:
            return v
        if len(v) > max_length:
            raise ValueError(
                f"Text is too long ({len(v)} characters); maximum is {max_length}"
            )
        return v


class ErrorResponse(BaseModel):
    status: str
    error: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> JSONResponse:
    """Liveness/readiness check. Never exposes filesystem paths."""
    try:
        get_config()
        config_ok = True
    except ConfigError as exc:
        logger.error("Health check failed: %s", exc)
        config_ok = False

    checks = {
        "config": config_ok,
        "engine": config_ok and engine_ready(),
        "encoder": config_ok and encoder_ready(),
        "auth": bool(API_KEY),
    }
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "service": "signbridge-service",
            "release": "phase-2",
            "checks": checks,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def _render_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ConfigError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, EncodingError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="Rendering failed.")


@app.post(
    "/v1/animations",
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
def create_animation(body: AnimRequest):
    """Render English text to a browser-safe H.264 MP4."""
    try:
        text = validate_text(body.text)
        with RENDER_LOCK:
            cfg = get_config()
            result = generate_animations(text, output_dir=cfg["output_dir"])
    except (ConfigError, EncodingError, ValueError) as exc:
        raise _render_error(exc) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Render failed for text: %r", body.text)
        raise _render_error(exc) from exc

    metadata = result.get("metadata") or {}
    width = metadata.get("width")
    height = metadata.get("height")
    nb_frames = metadata.get("nb_frames")
    fps = metadata.get("fps")
    duration = metadata.get("duration_sec")
    codec = metadata.get("codec")
    size = metadata.get("size_bytes") or result.get("size_bytes")
    filepath = result["filepath"]

    if not width or not height or not nb_frames or not fps:
        remove_output(filepath)
        raise HTTPException(
            status_code=500,
            detail=(
                "Generated file is not a valid MP4: "
                f"width={width}, height={height}, frames={nb_frames}, fps={fps}"
            ),
        )
    if str(codec or "").lower() != "h264":
        remove_output(filepath)
        raise HTTPException(
            status_code=503,
            detail=f"Generated file is not browser-safe H.264 (codec={codec!r}).",
        )

    logger.info(
        "Rendered text=%r filename=%s width=%s height=%s frames=%s dur=%s",
        text,
        result["filename"],
        width,
        height,
        nb_frames,
        duration,
    )

    return FileResponse(
        path=filepath,
        filename=result["filename"],
        media_type="video/mp4",
        background=BackgroundTask(remove_output, filepath),
        headers={
            "X-SignBridge-Size-Bytes": str(size),
            "X-SignBridge-Duration-Sec": f"{float(duration or 0.0):.3f}",
            "X-SignBridge-Width": str(width),
            "X-SignBridge-Height": str(height),
            "X-SignBridge-Fps": str(fps),
            "X-SignBridge-Frames": str(nb_frames),
            "X-SignBridge-Codec": str(codec),
        },
    )


@app.post(
    "/v1/animations/structured",
    status_code=200,
    dependencies=[Depends(require_api_key)],
)
def create_structured_pose(body: AnimRequest) -> JSONResponse:
    """Return the real SignBridge joint sequence for English text."""
    try:
        text = validate_text(body.text)
        with RENDER_LOCK:
            result = generate_pose_sequence(text)
    except (ConfigError, EncodingError, ValueError) as exc:
        raise _render_error(exc) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pose generation failed for text: %r", body.text)
        raise _render_error(exc) from exc

    logger.info(
        "Pose sequence text=%r glosses=%s frames=%s duration_ms=%s",
        text,
        result["glosses"],
        result["frameCount"],
        result["durationMs"],
    )
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Error handling (uniform JSON, no stack traces)
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            error=str(exc.detail),
            detail=None,
        ).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            error="internal_error",
            detail="An unexpected error occurred.",
        ).model_dump(),
    )
