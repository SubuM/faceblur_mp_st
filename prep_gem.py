"""Prepare and clean up the runtime model files for FaceBlur Studio.

``v17.py`` imports this module and calls :func:`prepare_models` when the app
starts. All model files live in ``models_runtime/`` next to this file. The app
registers :func:`cleanup_models` with :mod:`atexit`, so that folder is removed
when the Streamlit process exits.

Running this file directly downloads the models and leaves them in place::

    python prep_gem.py
"""

from __future__ import annotations

import atexit
import shutil
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models_runtime"

# Model filenames are also used by v17.py when it loads the prepared files.
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
BLAZE_FILENAME = "blaze_face_short_range.tflite"
SEGMENTER_FILENAME = "selfie_multiclass_256x256.tflite"
LANDMARKER_FILENAME = "face_landmarker.task"


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    urls: Sequence[str]
    min_bytes: int


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        YUNET_FILENAME,
        (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        100_000,
    ),
    ModelSpec(
        BLAZE_FILENAME,
        (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        ),
        100_000,
    ),
    ModelSpec(
        SEGMENTER_FILENAME,
        (
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
        ),
        1_000_000,
    ),
    ModelSpec(
        LANDMARKER_FILENAME,
        (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
        ),
        1_000_000,
    ),
)

_PREPARED = False
_PREPARE_LOCK = threading.Lock()
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_REGISTERED = False

ProgressCallback = Callable[[str], None]


def _looks_like_model(path: Path, min_bytes: int) -> bool:
    """Return whether ``path`` looks like a complete model file."""
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
        with path.open("rb") as handle:
            head = handle.read(64).lstrip().lower()
        # A captive portal / proxy error page is the usual failure mode.
        return not head.startswith((b"<html", b"<!doctype", b"{", b"[", b"404"))
    except OSError:
        return False


def _download(spec: ModelSpec, destination: Path) -> None:
    """Download one model, trying its mirrors until one succeeds."""
    partial = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    for url in spec.urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open(
                "wb"
            ) as handle:
                shutil.copyfileobj(response, handle)
            if _looks_like_model(partial, spec.min_bytes):
                partial.replace(destination)
                return
            raise RuntimeError(f"downloaded file is not a valid model: {url}")
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            last_error = exc
            partial.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not download {spec.filename} into {MODEL_DIR}. ({last_error})"
    )


def prepare_models(on_progress: ProgressCallback | None = None) -> Path:
    """Create the runtime model folder and download any missing models.

    The first call in a process does the work; later calls return immediately.
    Existing valid files are reused, so models downloaded manually with
    ``python prep_gem.py`` are not fetched again.
    """
    global _PREPARED
    if _PREPARED:
        return MODEL_DIR

    with _PREPARE_LOCK:
        if _PREPARED:
            return MODEL_DIR

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        for stale in MODEL_DIR.glob("*.part"):
            stale.unlink(missing_ok=True)

        for spec in MODELS:
            destination = MODEL_DIR / spec.filename
            if _looks_like_model(destination, spec.min_bytes):
                continue
            if on_progress is not None:
                on_progress(f"Downloading {spec.filename}…")
            _download(spec, destination)

        _PREPARED = True
        return MODEL_DIR


def is_prepared() -> bool:
    """Return whether this process has already prepared the models."""
    return _PREPARED


def model_path(filename: str) -> Path:
    """Return a prepared model file, raising if it is missing or invalid."""
    path = MODEL_DIR / filename
    spec = next((model for model in MODELS if model.filename == filename), None)
    if spec is not None and _looks_like_model(path, spec.min_bytes):
        return path
    if spec is None and path.is_file():
        return path
    raise RuntimeError(
        f"The required model {filename} is not available. Restart the app so "
        "prep_gem.py can prepare the models."
    )


def register_cleanup() -> None:
    """Delete the runtime model folder when the app process exits."""
    global _CLEANUP_REGISTERED
    with _CLEANUP_LOCK:
        if _CLEANUP_REGISTERED:
            return
        atexit.register(cleanup_models)
        _CLEANUP_REGISTERED = True


def cleanup_models() -> None:
    """Remove the runtime model folder and everything inside it."""
    shutil.rmtree(MODEL_DIR, ignore_errors=True)


def main() -> None:
    target = prepare_models(print)
    print(f"Models are ready in {target}")


if __name__ == "__main__":
    main()