"""animation_service.py -- SignBridge integration and headless rendering.

Service layer for the standalone SignBridge API. It reuses the ORIGINAL engine
without changing the animation-generation logic:

  * ``gen_pose()``                from ``<SIGNBRIDGE_ROOT>/main.py``
  * ``gen_vid()``                 from ``<SIGNBRIDGE_ROOT>/main.py``
  * ``AnimationDirector``         from ``<SIGNBRIDGE_ROOT>/animation_lib.py``

Design guarantees:

  * No ``os.chdir()`` -- every path is absolute, resolved from configuration.
  * The engine is imported lazily (once) with the repository root on
    ``sys.path``; importing this module does not load spaCy or PyTorch.
  * Rendering is headless: OpenCV GUI calls are neutralised and the engine is
    told ``preview=False``.
  * Output must be browser-safe H.264. A failed transcode is reported as an
    error; it is never silently downgraded to MPEG-4.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("signbridge-service")

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
API_DIR = Path(__file__).resolve().parent
REPO_ROOT = API_DIR.parent


class ConfigError(Exception):
    """Raised when a service configuration value is invalid/missing."""


class EncodingError(Exception):
    """Raised when a browser-safe H.264 file cannot be produced."""


# ---------------------------------------------------------------------------
# Environment (.env next to the repository root, or next to this module)
# ---------------------------------------------------------------------------
def _load_env_file(path: Path) -> None:
    """Parse a simple KEY=VALUE .env file into ``os.environ`` (no deps)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(REPO_ROOT / ".env")
_load_env_file(API_DIR / ".env")


# ---------------------------------------------------------------------------
# Configuration (validated lazily so /health can report a broken config)
# ---------------------------------------------------------------------------
_CONFIG: Optional[dict[str, Any]] = None
_CONFIG_LOCK = threading.Lock()


def _default_output_dir() -> str:
    if os.name == "nt":
        import tempfile

        return os.path.join(tempfile.gettempdir(), "signbridge-outputs")
    return "/tmp/signbridge-outputs"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def get_config() -> dict[str, Any]:
    """Return validated configuration (cached). Raises ConfigError."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    with _CONFIG_LOCK:
        if _CONFIG is not None:
            return _CONFIG

        root = os.path.abspath(
            os.environ.get("SIGNBRIDGE_ROOT") or str(REPO_ROOT)
        )
        if not os.path.isdir(root):
            raise ConfigError(f"SIGNBRIDGE_ROOT is not a directory: {root}")
        for required in ("main.py", "animation_lib.py"):
            if not os.path.isfile(os.path.join(root, required)):
                raise ConfigError(
                    f"SIGNBRIDGE_ROOT does not contain {required}: {root}"
                )
        pose_dir = os.path.join(root, "Pose")
        if not os.path.isdir(pose_dir):
            raise ConfigError(f"Pose directory is missing: {pose_dir}")

        output_dir = os.path.abspath(
            os.environ.get("OUTPUT_DIR") or _default_output_dir()
        )
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            raise ConfigError(
                f"OUTPUT_DIR cannot be created ({output_dir}): {exc}"
            ) from exc
        if not os.access(output_dir, os.W_OK):
            raise ConfigError(f"OUTPUT_DIR is not writable: {output_dir}")

        _CONFIG = {
            "signbridge_root": root,
            "pose_dir": pose_dir,
            "output_dir": output_dir,
            "max_text_length": _int_env("MAX_TEXT_LENGTH", 280),
            "max_request_size": _int_env("MAX_REQUEST_SIZE", 64 * 1024),
            "ffmpeg_timeout": _int_env("FFMPEG_TIMEOUT_SEC", 180),
        }
        return _CONFIG


# ---------------------------------------------------------------------------
# Headless adaptation (documented, minimal)
# ---------------------------------------------------------------------------
# The engine's preview is disabled by passing ``preview=False``; the module
# attributes are additionally neutralised as defence in depth. There is no
# per-request patch/unpatch, so concurrent requests cannot race on it.
import cv2  # noqa: E402


def _noop_cv2(*_args, **_kwargs):
    """No-op for cv2 GUI calls so rendering is always headless."""
    return 0


_CV2_PATCHED: dict[str, Any] = {}


def enable_headless_cv2() -> None:
    if _CV2_PATCHED:
        return
    for name in ("imshow", "waitKey", "destroyAllWindows", "namedWindow"):
        original = getattr(cv2, name, None)
        if original is None:
            continue
        _CV2_PATCHED[name] = original
        setattr(cv2, name, _noop_cv2)
    logger.info("OpenCV GUI calls neutralised (headless mode).")


def revert_cv2_preview() -> None:
    """Restore the original cv2 GUI functions (tests/tools only)."""
    while _CV2_PATCHED:
        name, original = _CV2_PATCHED.popitem()
        setattr(cv2, name, original)


if os.environ.get("SIGNBRIDGE_HEADLESS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
):
    enable_headless_cv2()


# ---------------------------------------------------------------------------
# Lazy engine import (thread-safe, absolute paths, repository root on sys.path)
# ---------------------------------------------------------------------------
_ENGINE_LOCK = threading.Lock()
_engine_main: Any = None
_pose_lib: Any = None


def _load_engine():
    """Import the original engine once. Returns ``(main_module, animation_lib)``."""
    global _engine_main, _pose_lib
    if _engine_main is not None:
        return _engine_main, _pose_lib

    with _ENGINE_LOCK:
        if _engine_main is not None:
            return _engine_main, _pose_lib

        cfg = get_config()
        root = Path(cfg["signbridge_root"]).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        # ``animation_lib`` is imported by the engine as a bare module, so it
        # must resolve from the repository root. Drop a stale copy if any.
        existing = sys.modules.get("animation_lib")
        if existing is not None:
            location = Path(getattr(existing, "__file__", "") or "").resolve()
            if not location.is_relative_to(root):
                logger.warning(
                    "Discarding animation_lib imported from %s", location
                )
                sys.modules.pop("animation_lib", None)

        import importlib

        pose_lib = importlib.import_module("animation_lib")
        location = Path(getattr(pose_lib, "__file__", "") or "").resolve()
        if not location.is_relative_to(root):
            raise ConfigError(
                f"animation_lib resolved outside SIGNBRIDGE_ROOT: {location}"
            )

        import importlib.util

        main_path = root / "main.py"
        spec = importlib.util.spec_from_file_location("signbridge_main", main_path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"Could not load the engine from {main_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["signbridge_main"] = module
        spec.loader.exec_module(module)

        for required in ("gen_pose", "gen_vid"):
            if not hasattr(module, required):
                raise ConfigError(
                    f"Engine {main_path} is missing {required}()"
                )

        _engine_main, _pose_lib = module, pose_lib
        logger.info("SignBridge engine loaded from %s", root)
        return _engine_main, _pose_lib


def engine_ready() -> bool:
    try:
        _load_engine()
        return True
    except Exception:  # noqa: BLE001
        logger.exception("Engine is not available")
        return False


# ---------------------------------------------------------------------------
# External tools (ffmpeg / ffprobe)
# ---------------------------------------------------------------------------
def _resolve_tool(env_name: str, default_name: str) -> str:
    raw = (os.environ.get(env_name) or default_name).strip()
    found = shutil.which(raw)
    if found:
        return found
    if os.path.isfile(raw) and os.access(raw, os.X_OK):
        return raw
    raise EncodingError(
        f"{default_name} is required for browser-safe H.264 output but was not "
        f"found (looked for {raw!r}). Install ffmpeg or set {env_name}."
    )


def encoder_ready() -> bool:
    try:
        _resolve_tool("FFMPEG_BIN", "ffmpeg")
        _resolve_tool("FFPROBE_BIN", "ffprobe")
        return True
    except EncodingError:
        return False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_text(text: str) -> str:
    """Validate and normalise user text for the SignBridge gloss engine."""
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    stripped = text.strip()
    if not stripped:
        raise ValueError("Text is required")
    max_length = get_config()["max_text_length"]
    if len(stripped) > max_length:
        raise ValueError(
            f"Text is too long ({len(stripped)} characters); maximum is {max_length}"
        )
    return stripped


# ---------------------------------------------------------------------------
# Video metadata (ffprobe)
# ---------------------------------------------------------------------------
def _ffprobe_json(filepath: str) -> dict[str, Any]:
    ffprobe = _resolve_tool("FFPROBE_BIN", "ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,codec_name",
        "-show_entries", "format=duration,size",
        "-of", "json",
        filepath,
    ]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=get_config()["ffmpeg_timeout"],
        )
    except OSError as exc:
        raise EncodingError(f"ffprobe could not run: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EncodingError("ffprobe timed out while reading the video") from exc

    if out.returncode != 0:
        raise EncodingError(
            f"ffprobe failed (exit {out.returncode}): {(out.stderr or '').strip()[:500]}"
        )
    try:
        return json.loads(out.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise EncodingError(f"ffprobe returned invalid JSON: {exc}") from exc


def get_video_metadata(filepath: str) -> dict[str, Any]:
    """Probe an MP4 and return width/height/nb_frames/fps/duration/codec."""
    info = _ffprobe_json(filepath)
    streams = info.get("streams") or []
    fmt = info.get("format") or {}
    stream = streams[0] if streams else {}

    width = stream.get("width")
    height = stream.get("height")
    nb_frames = stream.get("nb_frames")
    avg_frame_rate = stream.get("avg_frame_rate", "0/1")
    codec = stream.get("codec_name") or fmt.get("codec_name")

    def parse_fps(rate: str) -> float:
        try:
            num, den = rate.split("/")
            return float(num) / float(den) if float(den) else 0.0
        except Exception:
            return 0.0

    def parse_duration(value: Optional[str]) -> float:
        if value:
            try:
                return float(value)
            except ValueError:
                return 0.0
        try:
            n = int(nb_frames) if nb_frames else 0
            f = parse_fps(avg_frame_rate)
            return round(n / f, 3) if f else 0.0
        except Exception:
            return 0.0

    return {
        "width": width,
        "height": height,
        "nb_frames": nb_frames,
        "framerate": avg_frame_rate,
        "fps": round(parse_fps(avg_frame_rate), 3),
        "duration_sec": parse_duration(fmt.get("duration")),
        "codec": codec,
        "size_bytes": os.path.getsize(filepath),
        "file": os.path.basename(filepath),
    }


BROWSER_SAFE_CODECS = {"h264"}


def ensure_browser_codec(filepath: str) -> dict[str, Any]:
    """Transcode the engine's MPEG-4 output to H.264, or raise EncodingError.

    OpenCV's MP4 writer emits MPEG-4 Part 2 (``mpeg4``), which Chrome/Edge
    cannot decode in a ``<video>`` element. There is deliberately no fallback:
    returning a non-playable file as success is worse than failing the request.
    """
    metadata = get_video_metadata(filepath)
    codec = str(metadata.get("codec") or "").lower()
    if codec in BROWSER_SAFE_CODECS:
        return metadata

    ffmpeg = _resolve_tool("FFMPEG_BIN", "ffmpeg")
    temp_path = f"{filepath}.h264.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", filepath,
        "-an",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        temp_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=get_config()["ffmpeg_timeout"],
        )
        if result.returncode != 0:
            raise EncodingError(
                "ffmpeg failed to encode H.264 (exit "
                f"{result.returncode}): {(result.stderr or '').strip()[:500]}"
            )
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) <= 0:
            raise EncodingError("ffmpeg produced an empty H.264 file")
        os.replace(temp_path, filepath)
        logger.info("Transcoded animation to browser-safe H.264: %s", filepath)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    metadata = get_video_metadata(filepath)
    codec = str(metadata.get("codec") or "").lower()
    if codec not in BROWSER_SAFE_CODECS:
        raise EncodingError(
            f"Output is not browser-safe after transcoding (codec={codec!r})"
        )
    return metadata


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _unique_output(filepath: str) -> None:
    """Remove a partially written file (best effort)."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        pass


def generate_animations(text: str, output_dir: Optional[str] = None) -> dict[str, Any]:
    """Render English text to an H.264 MP4 animation.

    Returns a dict with ``filename``, ``filepath``, ``output_dir``, ``metadata``
    and ``status``. Raises ValueError for invalid input, ConfigError for a bad
    deployment, and EncodingError when H.264 output cannot be produced.
    """
    stripped = validate_text(text)
    cfg = get_config()

    stamp = int(time.time() * 1000)
    filename = f"{stamp}_{uuid.uuid4().hex}.mp4"
    out_dir = os.path.abspath(output_dir or cfg["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)

    engine, _pose_lib = _load_engine()

    try:
        gloss_sequence = engine.gen_pose(stripped)
        if not gloss_sequence:
            raise ValueError("No valid gloss generated from the provided text.")

        # Absolute output path + preview disabled: no cwd changes, no GUI calls.
        engine.gen_vid(gloss_sequence, filepath, preview=False)
    except Exception:
        _unique_output(filepath)
        raise

    if not os.path.exists(filepath):
        raise RuntimeError(f"SignBridge renderer failed to write output: {filepath}")
    if os.path.getsize(filepath) <= 0:
        _unique_output(filepath)
        raise RuntimeError(f"Generated file is empty: {filepath}")

    try:
        metadata = ensure_browser_codec(filepath)
    except EncodingError:
        _unique_output(filepath)
        raise

    return {
        "status": "ok",
        "filename": filename,
        "filepath": filepath,
        "output_dir": out_dir,
        "size_bytes": os.path.getsize(filepath),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Structured pose sequence (drives Signova's own SVG avatar)
# ---------------------------------------------------------------------------
POSE_FPS = 30


def generate_pose_sequence(text: str) -> dict[str, Any]:
    """Return the real SignBridge pose sequence for English text.

    Reuses the engine's gloss, loader, interpolation and transition logic; no
    video is rendered and no GUI is touched.
    """
    stripped = validate_text(text)
    cfg = get_config()
    engine, pose_lib = _load_engine()

    pose_dir = cfg["pose_dir"]
    buffer = StringIO()
    with redirect_stdout(buffer):
        gloss_sequence = engine.gen_pose(stripped)

    if not gloss_sequence:
        raise ValueError("No valid gloss generated from the provided text.")

    director = engine.AnimationDirector(
        poses_dir=pose_dir,
        output_file="",
        preview=False,
    )
    loader = director.loader

    segments: list[tuple[np.ndarray, str]] = []
    for name in gloss_sequence:
        path = pose_lib.resolve_pose_file(pose_dir, name)
        matrix, _joints = loader.load_sequence(path)
        if matrix is None or len(matrix) == 0:
            continue
        segments.append((matrix, name))

    if not segments:
        raise ValueError("No usable pose data was found for this text.")

    sequence_frames: list[tuple[np.ndarray, str]] = []
    last_frame: Optional[np.ndarray] = None
    for matrix, name in segments:
        if last_frame is not None:
            transition = director._create_transition(
                last_frame,
                matrix[0],
                director.transition_frames,
            )
            for frame in transition:
                sequence_frames.append((frame, "..."))
        for frame in matrix:
            sequence_frames.append((frame, name))
        last_frame = matrix[-1]

    joints = list(getattr(loader, "target_joints", []) or [])
    if not joints:
        raise ValueError("No joint data was found for this text.")
    joint_count = len(joints)

    frames: list[dict[str, Any]] = []
    for frame, pose_name in sequence_frames:
        packed: list[Optional[float]] = []
        for index in range(joint_count):
            x = float(frame[index][0])
            y = float(frame[index][1])
            if np.isnan(x) or np.isnan(y):
                packed.extend([None, None])
            else:
                packed.extend([round(x, 5), round(y, 5)])
        frames.append({"pose": pose_name, "p": packed})

    if not frames:
        raise ValueError("The pose sequence is empty.")

    dropped: list[str] = []
    dropped_letters: list[dict[str, str]] = []
    for line in buffer.getvalue().splitlines():
        match = re.search(r"Pose not available for:\s*(\S+)", line)
        if match:
            dropped.append(match.group(1))
            continue
        match = re.search(
            r"Letter\s+(\S+)\s+not found for name\s+(\S+)",
            line,
        )
        if match:
            dropped_letters.append(
                {"letter": match.group(1), "word": match.group(2)}
            )

    unique_dropped = list(dict.fromkeys(dropped))
    unique_letters: list[dict[str, str]] = []
    seen_letters: set[tuple[str, str]] = set()
    for item in dropped_letters:
        key = (item["letter"], item["word"])
        if key in seen_letters:
            continue
        seen_letters.add(key)
        unique_letters.append(item)

    duration_ms = round((len(frames) / POSE_FPS) * 1000)

    return {
        "status": "ok",
        "text": stripped,
        "glosses": [name for _matrix, name in segments],
        "droppedWords": unique_dropped,
        "droppedLetters": unique_letters,
        "fps": POSE_FPS,
        "durationMs": duration_ms,
        "frameCount": len(frames),
        "transitionFrames": director.transition_frames,
        "joints": joints,
        "frames": frames,
        "coordinateSpace": "signbridge_normalized",
        "source": "signbridge",
    }


# ---------------------------------------------------------------------------
# Output cleanup
# ---------------------------------------------------------------------------
def remove_output(filepath: str) -> None:
    """Delete a generated file (used after the response has been sent)."""
    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not delete %s: %s", filepath, exc)


def sweep_outputs(max_age_seconds: Optional[int] = None) -> int:
    """Delete stale MP4s left behind by a crash (best effort)."""
    try:
        cfg = get_config()
    except ConfigError:
        return 0
    if max_age_seconds is None:
        max_age_seconds = _int_env("OUTPUT_RETENTION_SEC", 24 * 3600)
    cutoff = time.time() - max_age_seconds
    removed = 0
    try:
        entries = os.scandir(cfg["output_dir"])
    except OSError:
        return 0
    with entries:
        for entry in entries:
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    os.remove(entry.path)
                    removed += 1
            except OSError:
                continue
    if removed:
        logger.info("Removed %d stale output file(s)", removed)
    return removed
