"""FaceBlur Studio — image/video face blurring, background editing and morphing.

This is ``v4.py`` extended with face morphing for images. The app first asks
whether the user is uploading an image, a video, or two images to morph, then
routes to the matching flow. The image and video flows share the detector, blur
and settings code. In image mode the background can optionally be removed
(transparent result) or replaced with a user-supplied background image: the
background keeps its native aspect ratio and the subject is scaled and placed to
match it.

The face-morph flow takes a source and a target image, finds the largest
face in each and determines its ROI in a roll-normalised frame (a tilted head
reports the same ROI as an upright one). The whole target image is resized
with a single uniform scale so its face ROI matches the source's in pixel
size, then the target face is placed *exactly* on the source face's footprint
— same size, position and tilt, anchored on the eye midpoint — so the mixed
face replaces the source face proportionately instead of being overlaid at a
compromise position. Both faces are warped to an intermediate shape with a
Delaunay-triangulated piecewise warp. The two warped faces are
cross-dissolved into a single mixed face — with the target's skin tone
matched to the source — and placed back on the source image inside a
feathered face oval, so the source framing and background are preserved. The
face can be located by MediaPipe's FaceLandmarker in one pass or by YuNet
first when the face is small in its photo, and the morph amount is
adjustable from 0% (source) to 100% (target). The blend can cover the face
only or the whole image.

The original app relied on APIs that no longer exist in the current, free
package releases:

* ``mediapipe.solutions.face_detection`` was removed in MediaPipe 0.10.31 and is
  absent from MediaPipe 1.x. Only the Tasks API (``mediapipe.tasks.python``)
  remains, and models are no longer bundled with the wheel.
* ``st.set_page_config(page_icon="timer_clock")`` is not a valid icon value.
* OpenCV 5 moved ``CascadeClassifier`` out of the core ``objdetect`` module.
* ``mp4v`` video files are not playable in browsers, so the download was fine
  but the Streamlit preview never worked.

This file keeps the same idea (upload a video or an image, blur every face,
download the result) but uses only currently maintained, completely free
components:

* Streamlit >= 1.50 for the UI (``st.status``, ``st.progress``, session state,
  ``download_button(on_click="ignore")`` so a download no longer re-runs the
  whole pipeline).
* OpenCV 5 + ``cv2.FaceDetectorYN`` (YuNet, BSD/MIT, ONNX) as the default
  detector — no ``opencv-contrib`` specific model is required and it is more
  accurate than BlazeFace for arbitrary videos.
* MediaPipe Tasks ``FaceDetector`` with ``blaze_face_short_range.tflite``
  (Apache-2.0) as an alternative backend. The model is downloaded once and
  cached, because MediaPipe stopped shipping models in the wheel.
* MediaPipe Tasks ``ImageSegmenter`` with ``selfie_multiclass_256x256.tflite``
  (Apache-2.0) for background removal/replacement in images. It is tuned for
  people (selfies and portraits) and runs locally on CPU.
* MediaPipe Tasks ``FaceLandmarker`` with ``face_landmarker.task`` (Apache-2.0)
  for face morphing. The 478-point face mesh is used to align the two faces and
  to build the piecewise-affine warp. The model is cached like the others.
* PyAV (LGPL FFmpeg bindings) to encode H.264/yuv420p — the only MP4 flavour
  that actually plays in browsers — and to copy the original audio track back
  in. ``imageio-ffmpeg`` is a drop-in alternative, but PyAV avoids spawning a
  subprocess per job.
* No paid APIs, no cloud calls, no API keys. Models come from Google Cloud
  Storage and the OpenCV Zoo and run locally on CPU.

Run with::

    pip install -r requirements-v1.txt
    streamlit run v5.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on the interpreter used
    raise ImportError(
        "OpenCV is not installed in the Python interpreter running this app:\n"
        f"  {sys.executable}\n"
        "Install the dependencies with:\n"
        f"  {sys.executable} -m pip install -r requirements-v1.txt"
    ) from exc

import numpy as np
import streamlit as st

try:  # PyAV gives us browser-playable H.264 and audio muxing.
    import av
except ImportError:  # pragma: no cover - optional fallback
    av = None  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_NAME = "FaceBlur Studio"
MIN_PYTHON = (3, 11)
MIN_STREAMLIT = (1, 50)

KIND_IMAGE = "Image"
KIND_VIDEO = "Video"
KIND_MORPH = "Face morph"

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff")
VIDEO_EXTENSIONS = ("mp4", "mov", "m4v", "avi", "mkv", "webm", "mpeg", "mpg")

IMAGE_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}

# JPEG has no alpha channel, so transparency is dropped for that format only.
IMAGE_ENCODERS = {
    "png": (".png", "PNG", [cv2.IMWRITE_PNG_COMPRESSION, 3]),
    "webp": (".webp", "WebP (quality 95)", [cv2.IMWRITE_WEBP_QUALITY, 95]),
    "bmp": (".bmp", "BMP", []),
    "tif": (".tiff", "TIFF", []),
    "tiff": (".tiff", "TIFF", []),
}
IMAGE_JPEG_ENCODER = (".jpg", "JPEG (quality 95)", [cv2.IMWRITE_JPEG_QUALITY, 95])

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
)
YUNET_MIN_BYTES = 100_000

BLAZE_FILENAME = "blaze_face_short_range.tflite"
BLAZE_URLS = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
)
BLAZE_MIN_BYTES = 100_000

SEGMENTER_FILENAME = "selfie_multiclass_256x256.tflite"
SEGMENTER_URLS = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
)
SEGMENTER_MIN_BYTES = 1_000_000

LANDMARKER_FILENAME = "face_landmarker.task"
LANDMARKER_URLS = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task",
)
LANDMARKER_MIN_BYTES = 1_000_000

DETECTOR_YUNET = "YuNet (OpenCV DNN)"
DETECTOR_MEDIAPIPE = "MediaPipe BlazeFace"
DETECTOR_LANDMARKER = "MediaPipe FaceLandmarker"

MORPH_FACE_ONLY = "Face only"
MORPH_WHOLE_IMAGE = "Whole image"

# How the face is located before the landmark mesh is fitted. The mesh itself
# always comes from MediaPipe; YuNet only steers where to look.
MORPH_FINDER_MEDIAPIPE = DETECTOR_LANDMARKER
MORPH_FINDER_YUNET = "YuNet (finds the face first)"

BLUR_GAUSSIAN = "Gaussian blur"
BLUR_PIXELATE = "Pixelate"
BLUR_SOLID = "Solid black"

QUALITY_CRF = {"High": 18, "Balanced": 23, "Small": 28}
QUALITY_PRESET = {"High": "medium", "Balanced": "veryfast", "Small": "veryfast"}

# Audio codecs that can be copied straight into an MP4 container.
MP4_AUDIO_CODECS = frozenset({"aac", "mp3", "mp2", "ac3", "eac3", "alac", "opus", "flac"})

ProgressCallback = Callable[[float, str], None]


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FaceBox:
    """An axis-aligned face box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float = 0.0


@dataclass
class ProcessStats:
    frames: int = 0
    detection_frames: int = 0
    frames_with_faces: int = 0
    faces: int = 0
    elapsed: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    detector: str = ""
    encoder: str = ""
    audio: str = "none"
    background: str = "none"
    morph: str = "none"


@dataclass
class ProcessingResult:
    data: bytes
    filename: str
    stats: ProcessStats
    settings: dict[str, object] = field(default_factory=dict)
    source: tuple[object, ...] = ("", 0)
    kind: str = KIND_VIDEO


# --------------------------------------------------------------------------- #
# Model management
# --------------------------------------------------------------------------- #


def _model_dirs() -> list[Path]:
    """Directories searched for model files, most specific first."""
    override = os.environ.get("FACEBLUR_MODEL_DIR")
    dirs = [Path.cwd() / "models"]
    if override:
        dirs.append(Path(override).expanduser())
    else:
        dirs.append(Path.home() / ".cache" / "faceblur" / "models")
    # De-duplicate while keeping order.
    unique: list[Path] = []
    for directory in dirs:
        if directory not in unique:
            unique.append(directory)
    return unique


def _looks_like_model(path: Path, min_bytes: int) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < min_bytes:
            return False
        with path.open("rb") as handle:
            head = handle.read(64).lstrip().lower()
        # A captive portal / proxy error page is the usual failure mode.
        return not head.startswith((b"<html", b"<!doctype", b"{", b"[", b"404"))
    except OSError:
        return False


def ensure_model(filename: str, urls: Sequence[str], min_bytes: int) -> Path:
    """Return a local copy of ``filename``, downloading it once if needed."""
    for directory in _model_dirs():
        candidate = directory / filename
        if _looks_like_model(candidate, min_bytes):
            return candidate

    target = _model_dirs()[-1] / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    # Unique partial name: two sessions may download the same model concurrently.
    partial = target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.part")

    last_error: Exception | None = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if _looks_like_model(partial, min_bytes):
                partial.replace(target)
                return target
            raise RuntimeError(f"downloaded file is not a valid model: {url}")
        except Exception as exc:  # noqa: BLE001 - we retry the next mirror
            last_error = exc
            partial.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not download {filename}. Download it manually into "
        f"{_model_dirs()[-1]} and try again. ({last_error})"
    )


# --------------------------------------------------------------------------- #
# Face detectors
# --------------------------------------------------------------------------- #


class FaceDetectorBackend(Protocol):
    name: str

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[FaceBox]:
        ...

    def close(self) -> None:
        ...


class YunetFaceDetector:
    """OpenCV's DNN face detector (YuNet). Free, tiny, and very fast on CPU."""

    name = DETECTOR_YUNET

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        max_side: int = 960,
    ) -> None:
        # Positional arguments keep this working on both OpenCV 4.8+ and 5.x.
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), float(score_threshold), float(nms_threshold), 5000
        )
        self._input_size: tuple[int, int] | None = None
        self._max_side = max_side

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[FaceBox]:
        height, width = frame_bgr.shape[:2]
        if height < 2 or width < 2:
            return []

        # Detect on a downscaled copy: much faster on 4K input, same accuracy.
        scale = min(1.0, self._max_side / max(height, width))
        if scale < 1.0:
            detect_frame = cv2.resize(
                frame_bgr,
                (max(2, round(width * scale)), max(2, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detect_frame = frame_bgr

        detect_h, detect_w = detect_frame.shape[:2]
        if self._input_size != (detect_w, detect_h):
            self._detector.setInputSize((detect_w, detect_h))
            self._input_size = (detect_w, detect_h)

        _, faces = self._detector.detect(detect_frame)
        if faces is None:
            return []

        inverse = 1.0 / scale
        boxes: list[FaceBox] = []
        for face in faces:
            x, y, w, h = (float(value) for value in face[:4])
            if w <= 1 or h <= 1:
                continue
            boxes.append(
                FaceBox(
                    x=round(x * inverse),
                    y=round(y * inverse),
                    w=round(w * inverse),
                    h=round(h * inverse),
                    score=float(face[-1]),
                )
            )
        return boxes

    def close(self) -> None:
        # cv2.FaceDetectorYN has no explicit release(); dropping the reference is enough.
        self._detector = None  # type: ignore[assignment]


class MediaPipeFaceDetector:
    """MediaPipe Tasks face detector (replaces the removed ``mp.solutions`` API)."""

    name = DETECTOR_MEDIAPIPE

    def __init__(self, model_path: Path, min_detection_confidence: float = 0.5) -> None:
        # Imported lazily: the module is heavy and only needed for this backend.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._vision = mp_vision
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=float(min_detection_confidence),
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        self._last_timestamp = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[FaceBox]:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp = max(int(timestamp_ms), self._last_timestamp + 1)
        self._last_timestamp = timestamp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._detector.detect_for_video(mp_image, timestamp)

        boxes: list[FaceBox] = []
        for detection in result.detections:
            bbox = detection.bounding_box
            if bbox.width <= 1 or bbox.height <= 1:
                continue
            score = float(detection.categories[0].score) if detection.categories else 0.0
            boxes.append(
                FaceBox(
                    x=int(bbox.origin_x),
                    y=int(bbox.origin_y),
                    w=int(bbox.width),
                    h=int(bbox.height),
                    score=score,
                )
            )
        return boxes

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


def build_detector(
    choice: str,
    confidence: float,
    progress: ProgressCallback | None = None,
) -> FaceDetectorBackend:
    """Create the requested detector, downloading its model on first use."""
    if choice == DETECTOR_MEDIAPIPE:
        if progress:
            progress(0.02, "Loading MediaPipe BlazeFace model…")
        model_path = ensure_model(BLAZE_FILENAME, BLAZE_URLS, BLAZE_MIN_BYTES)
        try:
            return MediaPipeFaceDetector(model_path, min_detection_confidence=confidence)
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Run `pip install -r requirements-v1.txt` "
                "or switch the detector to YuNet."
            ) from exc

    if progress:
        progress(0.02, "Loading YuNet model…")
    model_path = ensure_model(YUNET_FILENAME, YUNET_URLS, YUNET_MIN_BYTES)
    return YunetFaceDetector(model_path, score_threshold=confidence)


# --------------------------------------------------------------------------- #
# Background segmentation
# --------------------------------------------------------------------------- #


class BackgroundSegmenter:
    """MediaPipe Tasks selfie segmentation used for background editing.

    The multiclass model labels background, hair, body skin, face skin, clothes
    and accessories. Class 0 is the background, so ``1 - confidence[0]`` gives
    a soft foreground mask that already follows hair and clothing edges.
    """

    def __init__(self, model_path: Path) -> None:
        # Imported lazily: MediaPipe is heavy and only needed for this feature.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(
            mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_confidence_masks=True,
                output_category_mask=True,
            )
        )

    def foreground_alpha(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 mask: 0 is background, 1 is the subject."""
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._segmenter.segment(mp_image)

        alpha: np.ndarray | None = None
        if result.confidence_masks:
            background = result.confidence_masks[0].numpy_view()
            if background.ndim == 3:
                background = background[:, :, 0]
            alpha = 1.0 - background
        elif result.category_mask is not None:
            categories = result.category_mask.numpy_view()
            if categories.ndim == 3:
                categories = categories[:, :, 0]
            alpha = (categories != 0).astype(np.float32)
        if alpha is None:
            raise RuntimeError("The background segmentation model returned no mask.")

        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        if alpha.shape[:2] != (height, width):
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)

        # Remove low-confidence haze, then feather the cutout edge slightly so
        # hair and shoulders do not get a hard, jagged border.
        alpha = np.clip((alpha - 0.08) / 0.72, 0.0, 1.0)
        sigma = max(0.8, min(width, height) * 0.0015)
        return cv2.GaussianBlur(alpha, (0, 0), sigma)

    def close(self) -> None:
        try:
            self._segmenter.close()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


def build_segmenter(progress: ProgressCallback | None = None) -> BackgroundSegmenter:
    """Create the background segmenter, downloading its model on first use."""
    if progress:
        progress(0.02, "Loading background segmentation model…")
    model_path = ensure_model(SEGMENTER_FILENAME, SEGMENTER_URLS, SEGMENTER_MIN_BYTES)
    try:
        return BackgroundSegmenter(model_path)
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Run `pip install -r requirements-v1.txt`."
        ) from exc


# --------------------------------------------------------------------------- #
# Temporal smoothing
# --------------------------------------------------------------------------- #


class BoxSmoother:
    """Exponential smoothing of boxes matched across frames by IoU.

    Detectors jitter by a few pixels per frame; without smoothing the blur
    rectangle visibly crawls. Tracks are dropped as soon as a face disappears so
    blur is never painted where no face is.
    """

    def __init__(self, alpha: float = 0.45, iou_threshold: float = 0.2) -> None:
        self._alpha = alpha
        self._iou_threshold = iou_threshold
        self._tracks: list[FaceBox] = []

    @staticmethod
    def _iou(a: FaceBox, b: FaceBox) -> float:
        ax2, ay2 = a.x + a.w, a.y + a.h
        bx2, by2 = b.x + b.w, b.y + b.h
        inter_w = max(0, min(ax2, bx2) - max(a.x, b.x))
        inter_h = max(0, min(ay2, by2) - max(a.y, b.y))
        intersection = inter_w * inter_h
        if intersection == 0:
            return 0.0
        union = a.w * a.h + b.w * b.h - intersection
        return intersection / union if union else 0.0

    def update(self, boxes: Sequence[FaceBox]) -> list[FaceBox]:
        if not boxes:
            self._tracks = []
            return []
        if not self._tracks:
            self._tracks = list(boxes)
            return list(boxes)

        matched: dict[int, FaceBox] = {}
        claimed: set[int] = set()
        for index, box in enumerate(boxes):
            best_index, best_iou = -1, self._iou_threshold
            for track_index, track in enumerate(self._tracks):
                if track_index in claimed:
                    continue
                iou = self._iou(box, track)
                if iou > best_iou:
                    best_index, best_iou = track_index, iou
            if best_index >= 0:
                claimed.add(best_index)
                track = self._tracks[best_index]
                alpha = self._alpha
                matched[index] = FaceBox(
                    x=round(track.x + alpha * (box.x - track.x)),
                    y=round(track.y + alpha * (box.y - track.y)),
                    w=round(track.w + alpha * (box.w - track.w)),
                    h=round(track.h + alpha * (box.h - track.h)),
                    score=box.score,
                )

        smoothed = [matched.get(index, box) for index, box in enumerate(boxes)]
        self._tracks = smoothed
        return smoothed


# --------------------------------------------------------------------------- #
# Blurring
# --------------------------------------------------------------------------- #


def _odd(value: float, minimum: int = 3, maximum: int = 151) -> int:
    size = int(round(value))
    size = max(minimum, min(maximum, size))
    return size if size % 2 else size + 1


def _clip_roi(
    box: FaceBox, width: int, height: int, margin: float
) -> tuple[int, int, int, int] | None:
    pad_x = round(box.w * margin)
    pad_y = round(box.h * margin)
    x0 = max(0, box.x - pad_x)
    y0 = max(0, box.y - pad_y)
    x1 = min(width, box.x + box.w + pad_x)
    y1 = min(height, box.y + box.h + pad_y)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def apply_blur(
    frame: np.ndarray,
    boxes: Iterable[FaceBox],
    mode: str,
    strength: int,
    margin: float,
    feather: bool,
) -> int:
    """Blur/pixelate/black out every box in ``frame`` in place. Returns count."""
    height, width = frame.shape[:2]
    applied = 0
    intensity = max(0.0, min(1.0, strength / 100.0))

    for box in boxes:
        roi = _clip_roi(box, width, height, margin)
        if roi is None:
            continue
        x0, y0, x1, y1 = roi
        region = frame[y0:y1, x0:x1]
        roi_h, roi_w = region.shape[:2]

        if mode == BLUR_PIXELATE:
            block = max(2, round(min(roi_w, roi_h) * (0.02 + 0.10 * intensity)))
            tiny = cv2.resize(
                region,
                (max(1, roi_w // block), max(1, roi_h // block)),
                interpolation=cv2.INTER_AREA,
            )
            processed = cv2.resize(tiny, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        elif mode == BLUR_SOLID:
            processed = np.zeros_like(region)
        else:
            kernel = _odd(min(roi_w, roi_h) * (0.03 + 0.25 * intensity))
            processed = cv2.GaussianBlur(region, (kernel, kernel), 0)

        if feather:
            mask = np.zeros((roi_h, roi_w), dtype=np.float32)
            inset = max(1, round(min(roi_w, roi_h) * 0.15))
            cv2.rectangle(mask, (inset, inset), (roi_w - inset, roi_h - inset), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(roi_w, roi_h) * 0.06))
            alpha = mask[:, :, None]
            blended = processed.astype(np.float32) * alpha + region.astype(np.float32) * (1 - alpha)
            frame[y0:y1, x0:x1] = blended.astype(np.uint8)
        else:
            frame[y0:y1, x0:x1] = processed

        applied += 1

    return applied


# --------------------------------------------------------------------------- #
# Video writers
# --------------------------------------------------------------------------- #


class VideoSink(Protocol):
    encoder_name: str
    audio_status: str

    def write(self, frame_bgr: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...


def _packet_seconds(packet: "av.Packet") -> float | None:  # type: ignore[name-defined]
    timestamp = packet.pts if packet.pts is not None else packet.dts
    if timestamp is None:
        return None
    return float(timestamp * packet.time_base)


class AvH264Sink:
    """H.264 + yuv420p MP4 via PyAV, with the source audio track copied back in."""

    encoder_name = "H.264 (libx264)"

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        crf: int = 23,
        preset: str = "veryfast",
        audio_source: Path | None = None,
    ) -> None:
        if av is None:  # pragma: no cover - guarded by make_sink
            raise RuntimeError("PyAV is not installed")

        self._fps = fps if fps and np.isfinite(fps) and fps > 0 else 25.0
        self._container = av.open(str(path), mode="w", format="mp4")
        rate = Fraction(self._fps).limit_denominator(1001)
        self._stream = self._container.add_stream("libx264", rate=rate)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(int(crf)), "preset": preset}
        self._pts = 0

        self.audio_status = "none"
        self._audio_in = None
        self._audio_out = None
        self._audio_packets = None
        self._pending_audio = None

        if audio_source is not None:
            self._open_audio(audio_source)

    def _open_audio(self, audio_source: Path) -> None:
        try:
            self._audio_in = av.open(str(audio_source))
            if not self._audio_in.streams.audio:
                self.audio_status = "none"
                return
            source = self._audio_in.streams.audio[0]
            codec = source.codec_context.name
            if codec not in MP4_AUDIO_CODECS:
                self.audio_status = f"skipped ({codec} cannot be stored in MP4)"
                return
            self._audio_out = self._container.add_stream_from_template(source)
            self._audio_packets = self._audio_in.demux(source)
            self._pending_audio = next(self._audio_packets, None)
            self.audio_status = f"copied ({codec})"
        except Exception as exc:  # noqa: BLE001 - audio is best effort
            self.audio_status = f"skipped ({exc})"

    def _flush_audio(self, until_seconds: float) -> None:
        if self._audio_out is None or self._audio_packets is None:
            return
        while self._pending_audio is not None:
            packet = self._pending_audio
            seconds = _packet_seconds(packet)
            if seconds is not None and seconds > until_seconds:
                return
            if packet.dts is not None:
                packet.stream = self._audio_out
                self._container.mux(packet)
            self._pending_audio = next(self._audio_packets, None)

    def write(self, frame_bgr: np.ndarray) -> None:
        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        video_frame.pts = self._pts
        self._pts += 1
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)
        # Interleave audio roughly by timestamp so the result streams cleanly.
        self._flush_audio((self._pts - 1) / self._fps)

    def close(self) -> None:
        try:
            self._flush_audio(float("inf"))
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        finally:
            self._container.close()
            if self._audio_in is not None:
                self._audio_in.close()


class CvVideoSink:
    """Fallback writer used only when PyAV is unavailable.

    The ``mp4v`` codec is not playable in browsers, so the UI warns about it.
    """

    encoder_name = "MPEG-4 part 2 (OpenCV fallback)"
    audio_status = "none (install av to keep audio)"

    def __init__(self, path: Path, width: int, height: int, fps: float, **_: object) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError("OpenCV could not create the output video file.")

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def close(self) -> None:
        self._writer.release()


def make_sink(
    path: Path,
    width: int,
    height: int,
    fps: float,
    *,
    crf: int,
    preset: str,
    audio_source: Path | None,
) -> VideoSink:
    if av is not None:
        try:
            return AvH264Sink(
                path,
                width,
                height,
                fps,
                crf=crf,
                preset=preset,
                audio_source=audio_source,
            )
        except Exception:  # noqa: BLE001 - fall back to OpenCV if x264 is missing
            pass
    return CvVideoSink(path, width, height, fps)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def _open_capture(path: Path) -> cv2.VideoCapture:
    """Open a video, preferring the FFmpeg backend for widest codec support."""
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(
            "Could not open this video. The container or codec may be unsupported."
        )
    # Honour phone rotation metadata (no-op on builds without the flag).
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return capture


def _safe_fps(value: float) -> float:
    if not value or not np.isfinite(value) or value <= 0 or value > 240:
        return 25.0
    return float(value)


def _even(value: float) -> int:
    size = int(round(value))
    size -= size % 2
    return max(2, size)


def process_video(
    source: Path,
    destination: Path,
    detector: FaceDetectorBackend,
    *,
    blur_mode: str,
    strength: int,
    margin: float,
    feather: bool,
    smooth: bool,
    stride: int,
    max_width: int,
    crf: int,
    preset: str,
    keep_audio: bool,
    progress: ProgressCallback | None = None,
) -> ProcessStats:
    """Blur every face in ``source`` and write a browser-friendly MP4."""
    capture = _open_capture(source)
    sink: VideoSink | None = None
    stats = ProcessStats(detector=detector.name)
    started = time.perf_counter()

    try:
        fps = _safe_fps(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("No video frames could be decoded from this file.")

        source_height, source_width = frame.shape[:2]
        scale = 1.0
        if max_width and source_width > max_width:
            scale = max_width / source_width
        out_width = _even(source_width * scale)
        out_height = _even(source_height * scale)

        sink = make_sink(
            destination,
            out_width,
            out_height,
            fps,
            crf=crf,
            preset=preset,
            audio_source=source if keep_audio else None,
        )

        smoother = BoxSmoother() if smooth else None
        boxes: list[FaceBox] = []
        frame_index = 0
        stride = max(1, int(stride))

        while True:
            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(
                    frame, (out_width, out_height), interpolation=cv2.INTER_AREA
                )

            if frame_index % stride == 0:
                detected = detector.detect(frame, int(frame_index * 1000 / fps))
                boxes = smoother.update(detected) if smoother is not None else detected
                stats.detection_frames += 1
                stats.faces += len(boxes)
                if boxes:
                    stats.frames_with_faces += 1

            if boxes:
                apply_blur(frame, boxes, blur_mode, strength, margin, feather)

            sink.write(frame)
            frame_index += 1

            if progress is not None and (frame_index % 5 == 0 or frame_index == reported_frames):
                if reported_frames > 0:
                    fraction = min(0.97, 0.05 + 0.92 * frame_index / reported_frames)
                    progress(fraction, f"Blurring faces… {frame_index}/{reported_frames} frames")
                else:
                    progress(0.5, f"Blurring faces… {frame_index} frames")

            ok, frame = capture.read()
            if not ok:
                break

        if progress is not None:
            progress(0.97, f"Finalizing… {frame_index} frames encoded")

        stats.frames = frame_index
        stats.width = out_width
        stats.height = out_height
        stats.fps = fps
        stats.elapsed = time.perf_counter() - started
        stats.encoder = sink.encoder_name
        stats.audio = sink.audio_status
        return stats
    finally:
        capture.release()
        try:
            if sink is not None:
                sink.close()
        finally:
            # Always release the detector, even if finalising the file failed.
            detector.close()


# --------------------------------------------------------------------------- #
# Image pipeline
# --------------------------------------------------------------------------- #


def _image_encoder(extension: str) -> tuple[str, str, list[int]]:
    ext = extension.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return IMAGE_JPEG_ENCODER
    return IMAGE_ENCODERS.get(ext, IMAGE_ENCODERS["png"])


ALPHA_CAPABLE_EXTENSIONS = (".png", ".webp", ".tiff")


def _output_encoder(
    extension: str, *, remove_background: bool, replace_background: bool
) -> tuple[str, str, list[int]]:
    """Pick an encoder that can store the alpha channel when one is produced."""
    encode_ext, label, params = _image_encoder(extension)
    needs_alpha = remove_background and not replace_background
    if needs_alpha and encode_ext not in ALPHA_CAPABLE_EXTENSIONS:
        return IMAGE_ENCODERS["png"]
    return encode_ext, label, params


# Proportional replacement tuning. The subject is never blown up more than
# ``MAX_SUBJECT_UPSCALE`` times its cutout size, and the canvas stays within a
# usable resolution range so a tiny background does not produce a thumbnail.
MAX_SUBJECT_UPSCALE = 3.0
BACKGROUND_MAX_UPSCALE = 4.0
MIN_CANVAS_LONG_SIDE = 640
MAX_CANVAS_LONG_SIDE = 4096


def _decode_background(payload: bytes) -> np.ndarray:
    """Decode the replacement background at its native resolution."""
    background = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if background is None:
        raise RuntimeError("Could not decode the background image.")

    if background.dtype != np.uint8:
        peak = float(background.max()) if background.size else 255.0
        if peak > 255:
            background = np.clip(
                background.astype(np.float32) * (255.0 / peak), 0, 255
            ).astype(np.uint8)
        else:
            background = background.astype(np.uint8)

    if background.ndim == 2:
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    elif background.shape[2] == 2:
        background = cv2.cvtColor(background[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif background.shape[2] == 4:
        # Flatten the background's own transparency over white.
        alpha = background[:, :, 3:4].astype(np.float32) / 255.0
        foreground = background[:, :, :3].astype(np.float32)
        background = np.clip(
            foreground * alpha + 255.0 * (1.0 - alpha), 0, 255
        ).astype(np.uint8)

    if background.shape[0] == 0 or background.shape[1] == 0:
        raise RuntimeError("The background image is empty.")
    return background


def _subject_bounds(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the bounding box of the whole visible subject.

    Every meaningful foreground component is included, even a small detached
    hand, prop or thin edge, so the subject can be resized as a whole instead
    of being cropped. Only 1-3 pixel noise specks are ignored. Returns
    ``None`` when the mask has no usable foreground.
    """
    height, width = alpha.shape[:2]
    mask = (alpha > 0.02).astype(np.uint8)
    if not mask.any():
        mask = (alpha > 0.001).astype(np.uint8)
    if not mask.any():
        return None

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0, 0, width, height

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= 4
    if not keep.any():
        keep = areas == areas.max()
    kept = stats[1:][keep]

    x0 = int(kept[:, cv2.CC_STAT_LEFT].min())
    y0 = int(kept[:, cv2.CC_STAT_TOP].min())
    x1 = int((kept[:, cv2.CC_STAT_LEFT] + kept[:, cv2.CC_STAT_WIDTH]).max())
    y1 = int((kept[:, cv2.CC_STAT_TOP] + kept[:, cv2.CC_STAT_HEIGHT]).max())

    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def _subject_height_fraction(subject_aspect: float, canvas_aspect: float) -> float:
    """Choose how much of the canvas height the subject should occupy.

    A tall cutout (full body) fills more of the frame than a wide one
    (head-and-shoulders), which needs room around the head. The background
    orientation then nudges the result: wide scenes have room for a slightly
    larger subject, while tall scenes get a little more breathing room.
    """
    if subject_aspect < 0.55:
        fraction = 0.90  # full body
    elif subject_aspect < 1.05:
        fraction = 0.84  # waist-up
    else:
        fraction = 0.76  # head-and-shoulders

    if canvas_aspect >= 1.8:
        fraction = min(0.95, fraction * 1.04)
    elif canvas_aspect <= 0.8:
        fraction *= 0.97
    return fraction


def _limit_width(image: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale the final composite when a maximum width was requested."""
    if max_width and image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        target = (max_width, max(1, round(image.shape[0] * scale)))
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    return image


def _compose_on_background(
    image: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray,
    max_width: int = 0,
) -> np.ndarray:
    """Resize the whole subject onto the background, using it as the canvas.

    The background keeps its native aspect ratio and becomes the output frame.
    The subject layer is scaled as a whole -- never cropped to the largest
    connected component -- so detached hands, props and soft edges survive.
    It is centred horizontally and anchored near the bottom with a margin. The
    subject is never upscaled beyond ``MAX_SUBJECT_UPSCALE``; when that cap
    would leave it looking tiny, the background is scaled down instead so the
    subject-to-background proportion stays natural.
    """
    canvas = background
    canvas_height, canvas_width = canvas.shape[:2]

    # Keep the canvas in a usable size range before placing anything.
    long_side = max(canvas_width, canvas_height)
    if long_side < MIN_CANVAS_LONG_SIDE:
        factor = min(MIN_CANVAS_LONG_SIDE / long_side, BACKGROUND_MAX_UPSCALE)
    elif long_side > MAX_CANVAS_LONG_SIDE:
        factor = MAX_CANVAS_LONG_SIDE / long_side
    else:
        factor = 1.0
    if abs(factor - 1.0) > 1e-6:
        canvas_width = max(1, round(canvas_width * factor))
        canvas_height = max(1, round(canvas_height * factor))
        interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        canvas = cv2.resize(canvas, (canvas_width, canvas_height), interpolation=interpolation)

    bounds = _subject_bounds(alpha)
    if bounds is None:
        # Nothing survived segmentation: just return the replacement background.
        return _limit_width(canvas, max_width)

    x0, y0, x1, y1 = bounds
    subject_height = y1 - y0
    subject_width = x1 - x0
    subject_aspect = subject_width / subject_height
    canvas_aspect = canvas_width / canvas_height
    fraction = _subject_height_fraction(subject_aspect, canvas_aspect)

    # If the subject is small compared with the background, honouring the
    # natural fraction would require more than MAX_SUBJECT_UPSCALE. Shrink the
    # canvas instead, but never below the minimum usable long side.
    desired_scale = (canvas_height * fraction) / subject_height
    if desired_scale > MAX_SUBJECT_UPSCALE:
        wanted_height = (MAX_SUBJECT_UPSCALE * subject_height) / fraction
        shrink = wanted_height / canvas_height
        min_shrink = MIN_CANVAS_LONG_SIDE / max(canvas_width, canvas_height)
        shrink = min(max(shrink, min_shrink), 1.0)
        if shrink < 1.0 - 1e-6:
            canvas_width = max(1, round(canvas_width * shrink))
            canvas_height = max(1, round(canvas_height * shrink))
            canvas = cv2.resize(canvas, (canvas_width, canvas_height), interpolation=cv2.INTER_AREA)

    # Fit the whole subject inside the frame with a comfortable margin, then
    # anchor it near the bottom. The caps guarantee no visible edge is clipped.
    scale = (canvas_height * fraction) / subject_height
    scale = min(scale, MAX_SUBJECT_UPSCALE)
    scale = min(scale, (canvas_width * 0.90) / subject_width)
    scale = min(scale, (canvas_height * 0.92) / subject_height)
    scale = max(scale, 1e-6)

    target_width = max(1, round(subject_width * scale))
    target_height = max(1, round(subject_height * scale))

    # Resize a generous slice of the original layer instead of cropping the
    # subject itself. The extra padding is transparent, so every visible pixel
    # is preserved while we avoid scaling a huge empty canvas.
    pad_x = max(8, round(subject_width * 0.05))
    pad_y = max(8, round(subject_height * 0.05))
    crop_x0 = max(0, x0 - pad_x)
    crop_y0 = max(0, y0 - pad_y)
    crop_x1 = min(image.shape[1], x1 + pad_x)
    crop_y1 = min(image.shape[0], y1 + pad_y)

    layer = image[crop_y0:crop_y1, crop_x0:crop_x1]
    layer_alpha = alpha[crop_y0:crop_y1, crop_x0:crop_x1]
    scaled_width = max(1, round(layer.shape[1] * scale))
    scaled_height = max(1, round(layer.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    layer = cv2.resize(layer, (scaled_width, scaled_height), interpolation=interpolation)
    layer_alpha = cv2.resize(
        layer_alpha, (scaled_width, scaled_height), interpolation=interpolation
    )
    layer_alpha = np.clip(layer_alpha, 0.0, 1.0)

    # Position the scaled subject bounding box, then derive the layer offset.
    left = (canvas_width - target_width) // 2
    bottom_margin = max(1, round(canvas_height * 0.04))
    top = canvas_height - target_height - bottom_margin
    top = max(top, max(0, round(canvas_height * 0.04)))
    top = min(top, max(0, canvas_height - target_height))
    left = max(0, min(left, canvas_width - target_width))

    subject_left = round((x0 - crop_x0) * scale)
    subject_top = round((y0 - crop_y0) * scale)
    offset_x = left - subject_left
    offset_y = top - subject_top

    # Composite with clipping so transparent padding outside the canvas is
    # harmless. The subject itself always fits inside the frame.
    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    src_x1 = min(scaled_width, canvas_width - offset_x)
    src_y1 = min(scaled_height, canvas_height - offset_y)
    dst_x0 = max(0, offset_x)
    dst_y0 = max(0, offset_y)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        region = canvas[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
        subject = layer[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
        subject_alpha3 = layer_alpha[src_y0:src_y1, src_x0:src_x1][:, :, None]
        blended = subject * subject_alpha3 + region * (1.0 - subject_alpha3)
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(blended, 0, 255).astype(np.uint8)

    return _limit_width(canvas, max_width)


def process_image(
    payload: bytes,
    detector: FaceDetectorBackend,
    *,
    blur_mode: str,
    strength: int,
    margin: float,
    feather: bool,
    max_width: int,
    source_extension: str,
    remove_background: bool = False,
    background_image: bytes | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[bytes, ProcessStats]:
    """Blur every face in an encoded image and return the re-encoded bytes.

    The image is decoded with OpenCV, so every format the app accepts works the
    same way. A transparency channel, if present, is kept for formats that
    support it (PNG, WebP, TIFF) and dropped for JPEG.

    With ``remove_background`` the subject is segmented out. Passing
    ``background_image`` composites it behind the subject; without one the
    result is written with a transparent background (as PNG when the source
    format cannot store alpha).
    """
    stats = ProcessStats(detector=detector.name, frames=1)
    started = time.perf_counter()
    segmenter: BackgroundSegmenter | None = None

    try:
        if remove_background:
            segmenter = build_segmenter(progress)

        if progress is not None:
            progress(0.1, "Decoding image…")
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(
                "Could not decode this image. It may be corrupt or use an unsupported format."
            )

        # 16-bit PNG/TIFF inputs are supported but processed as 8-bit.
        if image.dtype != np.uint8:
            peak = float(image.max())
            if peak > 255:
                image = np.clip(image.astype(np.float32) * (255.0 / peak), 0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        alpha: np.ndarray | None = None
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 2:
            # Grayscale + alpha (some PNGs): keep the alpha plane.
            alpha = image[:, :, 1].copy()
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            alpha = image[:, :, 3].copy()
            image = image[:, :, :3].copy()

        source_height, source_width = image.shape[:2]
        if max_width and source_width > max_width:
            scale = max_width / source_width
            target = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
            image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, target, interpolation=cv2.INTER_AREA)

        subject_alpha: np.ndarray | None = None
        if remove_background:
            if progress is not None:
                progress(0.3, "Separating subject from background…")
            assert segmenter is not None
            subject_alpha = segmenter.foreground_alpha(image)

        if progress is not None:
            progress(0.45, "Detecting faces…")
        boxes = detector.detect(image)
        stats.detection_frames = 1
        stats.faces = len(boxes)
        stats.frames_with_faces = 1 if boxes else 0

        if progress is not None:
            progress(0.65, "Blurring faces…")
        apply_blur(image, boxes, blur_mode, strength, margin, feather)

        if remove_background and subject_alpha is not None:
            # Combine with any transparency the source image already had.
            if alpha is not None:
                subject_alpha = subject_alpha * (alpha.astype(np.float32) / 255.0)

            if background_image is not None:
                if progress is not None:
                    progress(0.8, "Fitting the subject to the new background…")
                background = _decode_background(background_image)
                image = _compose_on_background(image, subject_alpha, background, max_width)
                alpha = None
                stats.background = "replaced"
            else:
                alpha = np.round(subject_alpha * 255).astype(np.uint8)
                stats.background = "removed"
        elif alpha is not None:
            stats.background = "kept (original transparency)"

        encode_ext, encoder_label, params = _image_encoder(source_extension)
        if alpha is not None and encode_ext not in ALPHA_CAPABLE_EXTENSIONS:
            # JPEG/BMP cannot store transparency, so fall back to PNG.
            encode_ext, encoder_label, params = IMAGE_ENCODERS["png"]

        to_encode = image if alpha is None else np.dstack([image, alpha])
        ok, buffer = cv2.imencode(encode_ext, to_encode, params)
        if not ok:
            raise RuntimeError(f"Could not encode the result as {encoder_label}.")

        stats.width = image.shape[1]
        stats.height = image.shape[0]
        stats.elapsed = time.perf_counter() - started
        stats.encoder = encoder_label
        stats.audio = "n/a"
        if progress is not None:
            progress(1.0, "Done")
        return buffer.tobytes(), stats
    finally:
        if segmenter is not None:
            segmenter.close()
        detector.close()


# --------------------------------------------------------------------------- #
# Face morphing
# --------------------------------------------------------------------------- #

# Ordered loop around the face, used to build the feathered blend mask. These
# are the standard MediaPipe FaceMesh oval indices.
FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

# Iris centres (points 468-477 of the 478-point mesh) and the outer eye
# corners. They barely move with facial expression, so they anchor the
# roll-normalised face frame that the ROI size, the proportionate resize
# factor and the exact face placement are all derived from.
RIGHT_IRIS_INDEX = 468
LEFT_IRIS_INDEX = 473
EYE_CORNER_INDICES = (33, 133, 362, 263)


class FaceLandmarkerModel:
    """MediaPipe Tasks FaceLandmarker, used to find one face per image.

    The task returns a 478-point mesh (468 face points plus iris points). Only
    the largest face is used, so group photos still produce a predictable
    morph. Two finders can steer where the mesh is fitted:

    - ``MORPH_FINDER_MEDIAPIPE`` (default): the landmarker's own detector
      finds the face in one pass and is the most accurate when the face fills
      a fair share of the frame.
    - ``MORPH_FINDER_YUNET``: YuNet locates the face box first and the mesh
      is fitted on a crop normalised around it, which keeps working when the
      face is small relative to its photo.

    Whichever finder is chosen, the other one is tried as a fallback, so a
    borderline photo still yields landmarks whenever either detector can see
    the face.
    """

    name = DETECTOR_LANDMARKER

    def __init__(
        self, model_path: Path, finder: str = MORPH_FINDER_MEDIAPIPE
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self.finder = (
            finder if finder in (MORPH_FINDER_MEDIAPIPE, MORPH_FINDER_YUNET)
            else MORPH_FINDER_MEDIAPIPE
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=4,
            )
        )

    def largest_face(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        """Return pixel-space landmarks for the largest face in ``image_bgr``."""
        if self.finder == MORPH_FINDER_YUNET:
            points = self._landmark_via_yunet(image_bgr, label)
            if points is not None:
                return points
        try:
            return self._detect_largest(image_bgr, label)
        except RuntimeError:
            points = self._landmark_via_yunet(image_bgr, label)
            if points is not None:
                return points
            raise

    def _detect_largest(self, image_bgr: np.ndarray, label: str) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            raise RuntimeError(
                f"No face was detected in the {label} image. Use a clear, "
                "front-facing photo where the face is reasonably large."
            )

        def box_area(landmarks: Sequence[object]) -> float:
            xs = [point.x for point in landmarks]
            ys = [point.y for point in landmarks]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        face = max(result.face_landmarks, key=box_area)
        points = np.array([[point.x * width, point.y * height] for point in face], np.float32)
        if len(points) <= max(FACE_OVAL_INDICES):
            raise RuntimeError(
                "The face landmark model returned an unexpected number of points."
            )
        return points

    def _landmark_via_yunet(
        self, image_bgr: np.ndarray, label: str
    ) -> np.ndarray | None:
        """Locate the face with YuNet, then fit the mesh on a normalised crop.

        YuNet is a multi-scale sliding-window detector, so it finds the face
        box even when the face is small relative to its photo — exactly where
        the landmarker's own detector gives up. A generous crop around that
        box is rescaled so the face sits in the landmarker's detection sweet
        spot, the mesh is fitted there, and the points are mapped back into
        the original pixel space. Returns ``None`` when YuNet cannot find the
        face either.
        """
        try:
            model_path = ensure_model(YUNET_FILENAME, YUNET_URLS, YUNET_MIN_BYTES)
            detector = YunetFaceDetector(
                model_path, score_threshold=0.5, max_side=1920
            )
            try:
                boxes = detector.detect(image_bgr)
            finally:
                detector.close()
        except Exception:  # noqa: BLE001 - the caller keeps the original error
            return None
        if not boxes:
            return None

        box = max(boxes, key=lambda candidate: candidate.w * candidate.h)
        height, width = image_bgr.shape[:2]
        center_x = box.x + box.w / 2.0
        center_y = box.y + box.h / 2.0
        # Enough context that the face fills roughly a quarter of the crop,
        # which is squarely inside the landmarker's detection sweet spot.
        side = max(box.w, box.h) * 4.0
        x0 = max(0, int(center_x - side * 0.5))
        y0 = max(0, int(center_y - side * 0.5))
        x1 = min(width, int(center_x + side * 0.5))
        y1 = min(height, int(center_y + side * 0.5))
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None

        crop = image_bgr[y0:y1, x0:x1]
        long_side = max(crop.shape[:2])
        target_long_side = 900
        interpolation = (
            cv2.INTER_AREA if long_side > target_long_side else cv2.INTER_CUBIC
        )
        new_width = max(2, round(crop.shape[1] * target_long_side / long_side))
        new_height = max(2, round(crop.shape[0] * target_long_side / long_side))
        resized = cv2.resize(crop, (new_width, new_height), interpolation=interpolation)

        try:
            points = self._detect_largest(resized, label)
        except RuntimeError:
            return None

        scale_x = new_width / float(x1 - x0)
        scale_y = new_height / float(y1 - y0)
        mapped = np.stack(
            [points[:, 0] / scale_x + x0, points[:, 1] / scale_y + y0], axis=1
        )
        return mapped.astype(np.float32)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


def build_landmarker(
    progress: ProgressCallback | None = None,
    finder: str = MORPH_FINDER_MEDIAPIPE,
) -> FaceLandmarkerModel:
    """Create the face landmarker, downloading its model on first use."""
    if progress:
        progress(0.02, "Loading face landmark model…")
    model_path = ensure_model(LANDMARKER_FILENAME, LANDMARKER_URLS, LANDMARKER_MIN_BYTES)
    try:
        return FaceLandmarkerModel(model_path, finder=finder)
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Run `pip install -r requirements-v1.txt`."
        ) from exc


def _decode_image_bgr(payload: bytes, label: str) -> np.ndarray:
    """Decode an image for morphing and flatten any transparency over white."""
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(
            f"Could not decode the {label} image. It may be corrupt or use an "
            "unsupported format."
        )

    if image.dtype != np.uint8:
        peak = float(image.max()) if image.size else 255.0
        if peak > 255:
            image = np.clip(
                image.astype(np.float32) * (255.0 / peak), 0, 255
            ).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 2:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        foreground = image[:, :, :3].astype(np.float32)
        image = np.clip(
            foreground * alpha + 255.0 * (1.0 - alpha), 0, 255
        ).astype(np.uint8)
    return image


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack([points, np.ones((len(points), 1), np.float32)])
    return (homogeneous @ matrix.T).astype(np.float32)


def _face_frame(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Roll-normalised anchor, angle, width and height of a face mesh.

    The anchor is the midpoint between the eyes and the roll is the angle of
    the eye line, so neither moves with facial expression. The width and
    height are measured on the face oval *after* the roll is undone, which
    makes the ROI rotation-invariant: a tilted head reports exactly the same
    ROI size as an upright one. Every proportionate decision downstream —
    the resize factor and the placement transform — is derived from this
    frame.
    """
    if len(points) > LEFT_IRIS_INDEX:
        right_eye = points[RIGHT_IRIS_INDEX]
        left_eye = points[LEFT_IRIS_INDEX]
    else:
        # Meshes without iris points: the outer eye corners instead.
        corners = points[list(EYE_CORNER_INDICES)]
        right_eye = (corners[0] + corners[1]) / 2.0
        left_eye = (corners[2] + corners[3]) / 2.0
    anchor = (right_eye + left_eye) / 2.0
    roll = float(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))

    cos, sin = np.cos(-roll), np.sin(-roll)
    oval = points[list(FACE_OVAL_INDICES)] - anchor
    rotated = np.stack(
        [oval[:, 0] * cos - oval[:, 1] * sin, oval[:, 0] * sin + oval[:, 1] * cos],
        axis=1,
    )
    width = float(rotated[:, 0].max() - rotated[:, 0].min())
    height = float(rotated[:, 1].max() - rotated[:, 1].min())
    return anchor, roll, width, height


def _roi_scale_ratio(
    source_frame: tuple[np.ndarray, float, float, float],
    target_frame: tuple[np.ndarray, float, float, float],
) -> float:
    """Uniform, proportionate resize factor between two face ROIs.

    The mean of the width and height ratios keeps the resize a single uniform
    scale — no stretching — matching the ROI as closely as one scale can.
    """
    (_, _, source_width, source_height), (_, _, target_width, target_height) = (
        source_frame,
        target_frame,
    )
    if target_width <= 1e-6 or target_height <= 1e-6:
        return 1.0
    ratio = (source_width / target_width + source_height / target_height) / 2.0
    return float(ratio) if np.isfinite(ratio) and ratio > 1e-3 else 1.0


def _roi_similarity(
    source_frame: tuple[np.ndarray, float, float, float],
    target_frame: tuple[np.ndarray, float, float, float],
) -> np.ndarray:
    """Exact similarity transform placing the target ROI on the source ROI.

    Scale comes from the ROI size ratio, rotation from the eye-line angle
    difference, and translation maps the target eye midpoint exactly onto the
    source eye midpoint. Because the transform is derived directly from the
    two ROIs instead of being least-squares fitted to landmarks, it is not a
    compromise between conflicting points: the target face occupies precisely
    the source face's footprint — same size, same position, same tilt — which
    is what keeps the morph proportionate at every amount.
    """
    (source_anchor, source_roll, _, _), (target_anchor, target_roll, _, _) = (
        source_frame,
        target_frame,
    )
    scale = _roi_scale_ratio(source_frame, target_frame)
    angle = source_roll - target_roll
    a = scale * float(np.cos(angle))
    b = -scale * float(np.sin(angle))
    d = scale * float(np.sin(angle))
    e = scale * float(np.cos(angle))
    return np.array(
        [
            [a, b, source_anchor[0] - (a * target_anchor[0] + b * target_anchor[1])],
            [d, e, source_anchor[1] - (d * target_anchor[0] + e * target_anchor[1])],
        ],
        np.float32,
    )


def _face_roi(
    points: np.ndarray, image_shape: tuple[int, ...], margin: float = 0.15
) -> tuple[int, int, int, int]:
    """Bounding box of the face oval plus ``margin``, clamped to the image."""
    oval = points[list(FACE_OVAL_INDICES)]
    x_min, y_min = oval.min(axis=0)
    x_max, y_max = oval.max(axis=0)
    pad = margin * max(x_max - x_min, y_max - y_min)
    height, width = image_shape[:2]
    x0 = int(np.clip(np.floor(x_min - pad), 0, max(0, width - 2)))
    y0 = int(np.clip(np.floor(y_min - pad), 0, max(0, height - 2)))
    x1 = int(np.clip(np.ceil(x_max + pad), x0 + 1, width))
    y1 = int(np.clip(np.ceil(y_max + pad), y0 + 1, height))
    return x0, y0, x1, y1


def _align_target(
    target: np.ndarray,
    target_points: np.ndarray,
    source_points: np.ndarray,
    source_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Place the target face exactly on the source face's ROI.

    The similarity transform comes straight from the two ROIs
    (:func:`_roi_similarity`): a uniform scale from the ROI size ratio, the
    tilt difference of the two eye lines, and an exact eye-midpoint anchor.
    The target face therefore replaces the source face at the same size,
    position and tilt — not at a least-squares compromise position — so the
    morphed face stays proportionate to the source photo.
    """
    matrix = _roi_similarity(_face_frame(source_points), _face_frame(target_points))
    height, width = source_shape[:2]
    aligned = cv2.warpAffine(
        target,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, _transform_points(target_points, matrix)


def _triangulate(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Delaunay-triangulate points, returning index triples into ``points``."""
    points = np.asarray(points, np.float32)
    # Subdiv2D rejects exact duplicates, so triangulate the unique coordinates
    # and map each one back to its first original index.
    unique, inverse = np.unique(np.round(points, 3), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    first_index: dict[int, int] = {}
    for index, unique_index in enumerate(inverse):
        first_index.setdefault(int(unique_index), index)

    x, y, width, height = cv2.boundingRect(unique)
    subdiv = cv2.Subdiv2D((x - 10, y - 10, width + 20, height + 20))
    for point in unique:
        try:
            subdiv.insert((float(point[0]), float(point[1])))
        except cv2.error:  # pragma: no cover - duplicate/near-duplicate guard
            pass

    triangles: list[tuple[int, int, int]] = []
    for triangle in subdiv.getTriangleList():
        corners = triangle.reshape(3, 2)
        indices: list[int] = []
        valid = True
        for corner in corners:
            distances = np.linalg.norm(unique - corner, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > 1.0:
                valid = False
                break
            indices.append(first_index[nearest])
        if valid and len(set(indices)) == 3:
            triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def _warp_piecewise(
    image: np.ndarray,
    source_points: np.ndarray,
    destination_points: np.ndarray,
    triangles: Sequence[tuple[int, int, int]],
) -> np.ndarray:
    """Warp ``image`` triangle by triangle from source to destination points."""
    height, width = image.shape[:2]
    output = image.copy()
    for a, b, c in triangles:
        source_triangle = source_points[[a, b, c]].astype(np.float32)
        destination_triangle = destination_points[[a, b, c]].astype(np.float32)
        if abs(cv2.contourArea(destination_triangle)) < 0.5:
            continue

        x0 = max(0, int(np.floor(destination_triangle[:, 0].min())))
        y0 = max(0, int(np.floor(destination_triangle[:, 1].min())))
        x1 = min(width, int(np.ceil(destination_triangle[:, 0].max())) + 1)
        y1 = min(height, int(np.ceil(destination_triangle[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        matrix = cv2.getAffineTransform(source_triangle, destination_triangle)
        matrix[0, 2] -= x0
        matrix[1, 2] -= y0
        warped = cv2.warpAffine(
            image,
            matrix,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        polygon = np.round(destination_triangle - [x0, y0]).astype(np.int32)
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_8)
        # A tiny dilation closes one-pixel seams between adjacent triangles.
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

        region = output[y0:y1, x0:x1]
        region[mask > 0] = warped[mask > 0]
    return output


def _face_ring_points(points: np.ndarray, scale: float = 1.4) -> np.ndarray:
    """Return an expanded face-oval ring for warp coverage.

    The ring must reach beyond the feathered blend mask (the oval plus its
    blur halo) so the mixed face is fully covered by warped pixels.
    """
    oval = points[list(FACE_OVAL_INDICES)].astype(np.float32)
    center = oval.mean(axis=0, keepdims=True)
    return (center + (oval - center) * scale).astype(np.float32)


def _image_boundary_points(shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
            [width / 2, 0],
            [width - 1, height / 2],
            [width / 2, height - 1],
            [0, height / 2],
        ],
        np.float32,
    )


def _face_blend_mask(shape: tuple[int, ...], points: np.ndarray) -> np.ndarray:
    """Softly feathered mask covering the mixed face oval.

    The feather is proportional to the face size (about a tenth of its
    width), not to the image size, so the mixed face melts into the source
    photo instead of showing a visible paste border.
    """
    height, width = shape[:2]
    oval = points[list(FACE_OVAL_INDICES)].astype(np.float32)
    center = oval.mean(axis=0, keepdims=True)
    expanded = center + (oval - center) * 1.08
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 255)
    face_width = float(expanded[:, 0].max() - expanded[:, 0].min())
    sigma = max(1.0, face_width * 0.10)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigma)
    return np.clip(soft, 0.0, 1.0)


def _match_target_to_source_roi(
    target: np.ndarray,
    target_points: np.ndarray,
    scale: float,
    source_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize the target image so its face ROI matches the source face ROI.

    ``scale`` comes from :func:`_roi_scale_ratio` of the two face frames, so
    it is the uniform, proportionate size ratio of the two ROIs. Resizing the
    whole image with a quality interpolation keeps the later, near-rigid
    placement warp sharp: without this step a much smaller target face would
    be stretched into mush and a much larger one would waste resolution. When
    a full-image resize would explode (a tiny face inside a huge photo), a
    generous crop around the face ROI is resized instead so the intermediate
    stays bounded. The landmarks are returned in the new pixel space.
    """
    if not np.isfinite(scale) or scale <= 0.0 or abs(scale - 1.0) < 1e-3:
        return target, target_points

    height, width = target.shape[:2]
    projected_long_side = max(width, height) * scale
    bound = max(4096, max(source_shape[:2]) * 2)

    crop = None
    if projected_long_side > bound:
        # Tiny face in a huge photo: keep a generous region around the face
        # but never so much context that the resized crop exceeds the bound.
        roi_x0, roi_y0, roi_x1, roi_y1 = _face_roi(
            target_points, target.shape, margin=0.0
        )
        roi_long_side = max(roi_x1 - roi_x0, roi_y1 - roi_y0)
        margin = 1.0
        if roi_long_side * scale * (1.0 + 2.0 * margin) > bound:
            margin = max(0.1, (bound / (scale * roi_long_side) - 1.0) / 2.0)
        crop = _face_roi(target_points, target.shape, margin=margin)

    if crop is not None:
        x0, y0, x1, y1 = crop
        working = target[y0:y1, x0:x1]
        points = target_points - np.array([x0, y0], np.float32)
    else:
        working, points = target, target_points

    new_width = max(2, round(working.shape[1] * scale))
    new_height = max(2, round(working.shape[0] * scale))
    if abs(new_width - working.shape[1]) <= 1 and abs(new_height - working.shape[0]) <= 1:
        return working, points

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(working, (new_width, new_height), interpolation=interpolation)
    return resized, points * scale


def _match_face_colors(
    reference: np.ndarray, to_adjust: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Nudge the colours of ``to_adjust`` toward ``reference`` inside the face.

    Both images must already be warped to the same face shape, so the same
    oval covers comparable skin on each. Per-channel medians of that region
    give a robust tone estimate; the gain is clipped so extreme differences
    (glasses, heavy make-up, stage lighting) cannot blow the image out. This
    is what keeps the mixed face from looking pasted onto the source.
    """
    height, width = reference.shape[:2]
    oval = points[list(FACE_OVAL_INDICES)].astype(np.float32)
    center = oval.mean(axis=0, keepdims=True)
    inner = center + (oval - center) * 0.75
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 255)
    if int(mask.sum()) == 0:
        return to_adjust

    gains: list[float] = []
    for channel in range(3):
        ref_median = float(np.median(reference[:, :, channel][mask > 0]))
        adj_median = float(np.median(to_adjust[:, :, channel][mask > 0]))
        if ref_median < 2.0 or adj_median < 2.0:
            gains.append(1.0)
            continue
        gains.append(min(1.8, max(0.55, ref_median / adj_median)))

    adjusted = to_adjust.astype(np.float32)
    for channel, gain in enumerate(gains):
        adjusted[:, :, channel] *= gain
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def process_morph_image(
    source_payload: bytes,
    target_payload: bytes,
    *,
    amount: float,
    area: str,
    max_width: int,
    source_extension: str,
    finder: str = MORPH_FINDER_MEDIAPIPE,
    progress: ProgressCallback | None = None,
) -> tuple[bytes, ProcessStats]:
    """Mix the largest faces of the source and target into one blended face.

    ``amount`` runs from 0 (source) to 1 (target). The pipeline is:

    1. Landmark the largest face of each image and determine its ROI in a
       roll-normalised frame (a tilted head reports the same ROI as an
       upright one).
    2. Resize the whole target image with a single uniform, proportionate
       scale so its face ROI matches the source face ROI in pixel size.
    3. Place the resized target face *exactly* on the source face's
       footprint — same size, position and tilt, anchored on the eye
       midpoint — derived directly from the two ROIs, not fitted.
    4. Warp both faces to the intermediate face shape with a Delaunay-
       triangulated piecewise-affine warp and cross-dissolve them, with the
       target's skin tone matched to the source.
    5. Place that mixed face back on the source image inside a feathered
       face oval, so the source background and framing are preserved.

    ``area`` limits the blend to the face (default) or applies it to the
    whole aligned image. ``finder`` picks how the face is located:
    ``MORPH_FINDER_MEDIAPIPE`` (one pass) or ``MORPH_FINDER_YUNET`` (YuNet
    locates the box first, which is more reliable for small faces).
    """
    if finder not in (MORPH_FINDER_MEDIAPIPE, MORPH_FINDER_YUNET):
        finder = MORPH_FINDER_MEDIAPIPE
    stats = ProcessStats(
        detector=(
            f"{DETECTOR_YUNET} + {DETECTOR_LANDMARKER}"
            if finder == MORPH_FINDER_YUNET
            else DETECTOR_LANDMARKER
        ),
        frames=1,
    )
    started = time.perf_counter()
    landmarker: FaceLandmarkerModel | None = None

    try:
        amount = float(np.clip(amount, 0.0, 1.0))
        if area not in (MORPH_FACE_ONLY, MORPH_WHOLE_IMAGE):
            area = MORPH_FACE_ONLY

        landmarker = build_landmarker(progress, finder=finder)

        if progress is not None:
            progress(0.15, "Decoding images…")
        source = _decode_image_bgr(source_payload, "source")
        target = _decode_image_bgr(target_payload, "target")
        if max_width:
            source = _limit_width(source, max_width)
            target = _limit_width(target, max_width)

        if progress is not None:
            progress(0.3, "Detecting faces…")
        source_points = landmarker.largest_face(source, "source")
        target_points = landmarker.largest_face(target, "target")
        stats.faces = 2

        # Roll-normalised ROI frame of each face. The proportionate resize
        # factor and the exact placement transform both derive from these.
        source_frame = _face_frame(source_points)
        target_frame = _face_frame(target_points)
        resize_factor = _roi_scale_ratio(source_frame, target_frame)
        (_, _, source_roi_w, source_roi_h) = source_frame
        (_, _, target_roi_w, target_roi_h) = target_frame

        if amount <= 1e-4:
            result = source.copy()
        else:
            # Bring the target to the source's face resolution before any
            # warping happens, so neither face is starved or blown up.
            if progress is not None:
                progress(0.4, "Matching the target face ROI to the source…")
            target, target_points = _match_target_to_source_roi(
                target, target_points, resize_factor, source.shape
            )

            if progress is not None:
                progress(0.5, "Placing the target face on the source face…")
            aligned_target, aligned_points = _align_target(
                target, target_points, source_points, source.shape
            )
            morph_points = (
                (1.0 - amount) * source_points + amount * aligned_points
            ).astype(np.float32)

            if progress is not None:
                progress(0.6, "Warping faces…")
            if area == MORPH_FACE_ONLY:
                source_all = np.vstack([source_points, _face_ring_points(source_points)])
                target_all = np.vstack([aligned_points, _face_ring_points(aligned_points)])
                morph_all = np.vstack([morph_points, _face_ring_points(morph_points)])
            else:
                boundary = _image_boundary_points(source.shape)
                source_all = np.vstack([source_points, boundary])
                target_all = np.vstack([aligned_points, boundary])
                morph_all = np.vstack([morph_points, boundary])
            triangles = _triangulate(morph_all)
            warped_source = _warp_piecewise(source, source_all, morph_all, triangles)
            warped_target = _warp_piecewise(
                aligned_target, target_all, morph_all, triangles
            )

            if progress is not None:
                progress(0.8, "Blending faces…")
            # Match the target's tone to the source inside the face region so
            # the mixed face looks native to the source photo.
            warped_target = _match_face_colors(warped_source, warped_target, morph_points)

            if area == MORPH_FACE_ONLY:
                # Cross-dissolve the two warped faces into the mixed face.
                # The target's share fades out with the mask, so none of the
                # target's background leaks in around the face; then the mixed
                # face is placed on the source image inside the feathered
                # oval, leaving the source background and framing untouched.
                mask = _face_blend_mask(source.shape, morph_points)
                target_weight = mask * amount
                mixed = (
                    warped_source.astype(np.float32) * (1.0 - target_weight)[:, :, None]
                    + warped_target.astype(np.float32) * target_weight[:, :, None]
                )
                blended = (
                    source.astype(np.float32) * (1.0 - mask)[:, :, None]
                    + mixed * mask[:, :, None]
                )
            else:
                blended = (
                    warped_source.astype(np.float32) * (1.0 - amount)
                    + warped_target.astype(np.float32) * amount
                )
            result = np.clip(blended, 0, 255).astype(np.uint8)

        if progress is not None:
            progress(0.92, "Encoding result…")
        encode_ext, encoder_label, params = _image_encoder(source_extension)
        ok, buffer = cv2.imencode(encode_ext, result, params)
        if not ok:
            raise RuntimeError(f"Could not encode the result as {encoder_label}.")

        stats.morph = (
            f"{area} · {round(amount * 100)}% · target ROI "
            f"{target_roi_w:.0f}×{target_roi_h:.0f} px resized ×{resize_factor:.2f} "
            f"onto source ROI {source_roi_w:.0f}×{source_roi_h:.0f} px"
        )
        stats.width = result.shape[1]
        stats.height = result.shape[0]
        stats.elapsed = time.perf_counter() - started
        stats.encoder = encoder_label
        stats.audio = "n/a"
        if progress is not None:
            progress(1.0, "Done")
        return buffer.tobytes(), stats
    finally:
        if landmarker is not None:
            landmarker.close()


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(text).split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _check_runtime() -> bool:
    """Fail loudly when an old Python/Streamlit picks up this file.

    A very common failure is `streamlit` on PATH belonging to a different
    interpreter than the one the dependencies were installed into.
    """
    problems: list[str] = []
    if sys.version_info < MIN_PYTHON:
        problems.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is too old "
            f"(needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+). Interpreter: {sys.executable}"
        )
    if _parse_version(st.__version__) < MIN_STREAMLIT:
        problems.append(
            f"Streamlit {st.__version__} is too old "
            f"(needs {MIN_STREAMLIT[0]}.{MIN_STREAMLIT[1]}+). "
            "A different Streamlit is probably first on your PATH."
        )
    if problems:
        st.error(
            "This app cannot run in the current environment:\n\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + "\n\nStart it with the project's virtual environment instead:\n\n"
            "```\n./.venv/bin/streamlit run v5.py\n```"
        )
        return False
    return True


def _page_config() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                f"{APP_NAME} runs entirely on your machine. "
                "YuNet (OpenCV Zoo), MediaPipe BlazeFace, MediaPipe selfie "
                "segmentation and MediaPipe FaceLandmarker models are free to use."
            )
        },
    )


def _sidebar(kind: str) -> dict[str, object]:
    is_video = kind == KIND_VIDEO
    is_morph = kind == KIND_MORPH
    with st.sidebar:
        if is_morph:
            st.header("Morph")
            amount = st.slider(
                "Morph amount",
                min_value=0,
                max_value=100,
                value=50,
                format="%d%%",
                help=(
                    "0% keeps the source face, 100% uses the target face. "
                    "Values in between blend the two shapes and colours."
                ),
            )
            area = st.radio(
                "Blend area",
                (MORPH_FACE_ONLY, MORPH_WHOLE_IMAGE),
                help=(
                    "Face only keeps the source background and blends just the "
                    "face region. Whole image also cross-dissolves the aligned "
                    "target background, which is best when both photos have a "
                    "similar composition."
                ),
            )
            finder = st.radio(
                "Face finder",
                (MORPH_FINDER_MEDIAPIPE, MORPH_FINDER_YUNET),
                help=(
                    "How the face is located before morphing. MediaPipe "
                    "FaceLandmarker finds the face and its 478-point mesh in one "
                    "pass. YuNet first locates the face box and the mesh is then "
                    "fitted on an optimally sized crop — more reliable when the "
                    "face is small in the photo. Either way the mesh always "
                    "comes from MediaPipe, and the other finder is tried as a "
                    "fallback."
                ),
            )

            st.divider()
            st.header("Output")
            width_choice = st.selectbox(
                "Maximum width",
                ("Original", "1920 px", "1280 px", "854 px"),
                help="Downscales the saved image. Original keeps the full resolution.",
            )

            st.divider()
            st.caption(
                "100% local processing. No uploads to third-party services, no API keys. "
                f"Faces are found with {finder}."
            )
            return {
                "max_width": 0 if width_choice == "Original" else int(width_choice.split()[0]),
                "morph_amount": amount / 100.0,
                "morph_area": area,
                "morph_finder": finder,
            }

        st.header("Detection")
        detector = st.radio(
            "Face detector",
            (DETECTOR_YUNET, DETECTOR_MEDIAPIPE),
            help=(
                "YuNet is a tiny ONNX model that handles faces at any distance. "
                "MediaPipe BlazeFace is the modern Tasks API replacement for the "
                "removed mp.solutions detector and is tuned for close-range faces."
            ),
        )
        confidence = st.slider(
            "Detection confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.6 if detector == DETECTOR_YUNET else 0.5,
            step=0.05,
            help="Lower it to catch more faces (and more false positives).",
        )
        if is_video:
            stride = st.select_slider(
                "Detect every N frames",
                options=(1, 2, 3, 4),
                value=1,
                help="Detections are reused between frames, so 2–3 is much faster with little quality loss.",
            )

        st.divider()
        st.header("Blur")
        blur_mode = st.selectbox("Style", (BLUR_GAUSSIAN, BLUR_PIXELATE, BLUR_SOLID))
        strength = st.slider("Strength", 1, 100, 55)
        margin = st.slider(
            "Box padding",
            0,
            50,
            12,
            format="%d%%",
            help="Expands each face box so ears, chin and hairline are covered too.",
        )
        feather = st.checkbox(
            "Soft edges",
            value=False,
            help="Feathers the blur border. Slightly prettier, marginally less coverage.",
        )
        if is_video:
            smooth = st.checkbox(
                "Temporal smoothing",
                value=True,
                help="Reduces the frame-to-frame jitter of the blur rectangle.",
            )

        if not is_video:
            st.divider()
            st.header("Background")
            remove_background = st.checkbox(
                "Remove background",
                value=False,
                help=(
                    "Separates the subject with MediaPipe's selfie segmentation model "
                    "(best on photos of people). Upload a background image below the "
                    "uploader to replace it; otherwise the background becomes transparent. "
                    "With a replacement background, the subject is scaled and placed to "
                    "match the background automatically."
                ),
            )

        st.divider()
        st.header("Output")
        if is_video:
            quality = st.selectbox("Quality", tuple(QUALITY_CRF), index=1)
        width_choice = st.selectbox(
            "Maximum width",
            ("Original", "1920 px", "1280 px", "854 px"),
            help=(
                "Downscaling makes processing much faster for 4K footage."
                if is_video
                else "Downscales the saved image. Original keeps the full resolution."
            ),
        )
        if is_video:
            keep_audio = st.checkbox("Keep original audio", value=True)

        st.divider()
        st.caption(
            "100% local processing. No uploads to third-party services, no API keys. "
            f"PyAV encoder: {'available' if av is not None else 'missing (install av)'}."
        )

    settings: dict[str, object] = {
        "detector": detector,
        "confidence": confidence,
        "blur_mode": blur_mode,
        "strength": strength,
        "margin": margin / 100.0,
        "feather": feather,
        "max_width": 0 if width_choice == "Original" else int(width_choice.split()[0]),
    }
    if is_video:
        settings.update(
            stride=stride,
            smooth=smooth,
            quality=quality,
            keep_audio=keep_audio,
        )
    else:
        settings["remove_background"] = remove_background
    return settings


def _render_stats(result: ProcessingResult) -> None:
    stats = result.stats
    if result.kind in (KIND_IMAGE, KIND_MORPH):
        is_morph = result.kind == KIND_MORPH
        columns = st.columns(3)
        columns[0].metric(
            "Faces morphed" if is_morph else "Faces blurred", f"{stats.faces:,}"
        )
        columns[1].metric("Resolution", f"{stats.width}×{stats.height}")
        columns[2].metric("Processing time", f"{stats.elapsed:.2f}s")
        caption = f"{stats.detector} · {stats.encoder}"
        if is_morph and stats.morph != "none":
            caption += f" · morph: {stats.morph}"
        elif stats.background != "none":
            background_label = (
                "replaced (auto-fit)" if stats.background == "replaced" else stats.background
            )
            caption += f" · background: {background_label}"
        st.caption(caption)
        if not is_morph and stats.faces == 0:
            st.warning(
                "No faces were detected, so the image was only re-encoded. "
                "Try lowering the detection confidence or switching detectors."
            )
        return

    face_ratio = (stats.frames_with_faces / stats.frames * 100) if stats.frames else 0.0
    columns = st.columns(4)
    columns[0].metric("Frames", f"{stats.frames:,}")
    columns[1].metric("Faces blurred", f"{stats.faces:,}")
    columns[2].metric("Frames with a face", f"{face_ratio:.0f}%")
    columns[3].metric("Processing time", f"{stats.elapsed:.1f}s")

    st.caption(
        f"{stats.width}×{stats.height} · {stats.fps:.2f} fps · "
        f"{stats.detector} · {stats.encoder} · audio: {stats.audio}"
    )

    if stats.faces == 0:
        st.warning(
            "No faces were detected, so the video was only re-encoded. "
            "Try lowering the detection confidence or switching detectors."
        )
    if stats.encoder.startswith("MPEG-4"):
        st.warning(
            "PyAV is not installed, so the output uses MPEG-4 part 2. "
            "It downloads fine but will not play in a browser. "
            "Install `av` for H.264 output."
        )


def _render_result(result: ProcessingResult) -> None:
    st.subheader("Result")
    if result.kind in (KIND_IMAGE, KIND_MORPH):
        left, right = st.columns(2)
        with left:
            st.image(result.data, caption=result.filename)
        with right:
            extension = Path(result.filename).suffix.lower().lstrip(".")
            st.download_button(
                "Download result",
                data=result.data,
                file_name=result.filename,
                mime=IMAGE_MIME.get(extension, "application/octet-stream"),
                type="primary",
                # Do not rerun the script just because the user clicked download.
                on_click="ignore",
            )
            st.caption(f"{len(result.data) / 1e6:.2f} MB · {result.filename}")
        _render_stats(result)
        with st.expander("Settings used"):
            st.json(result.settings)
        return

    left, right = st.columns(2)
    if result.stats.encoder.startswith("H.264"):
        with right:
            st.video(result.data, format="video/mp4")
    else:
        with right:
            st.info("Preview needs H.264 output (install `av`). The file is still downloadable.")

    with left:
        st.download_button(
            "Download blurred video",
            data=result.data,
            file_name=result.filename,
            mime="video/mp4",
            type="primary",
            # Do not rerun the script just because the user clicked download.
            on_click="ignore",
        )
        st.caption(f"{len(result.data) / 1e6:.1f} MB · {result.filename}")

    _render_stats(result)

    with st.expander("Settings used"):
        st.json(result.settings)


def _run_video(uploaded: st.runtime.uploaded_file_manager.UploadedFile, settings: dict[str, object]) -> None:
    payload = uploaded.getvalue()
    stem = Path(uploaded.name).stem or "video"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_ ").strip() or "video"
    output_name = f"{safe_stem}_blurred.mp4"

    progress_bar = st.progress(0.0, text="Starting…")
    status = st.status("Processing video…", expanded=True)

    def report(fraction: float, message: str) -> None:
        progress_bar.progress(max(0.0, min(1.0, fraction)), text=message)
        status.update(label=message)

    try:
        with tempfile.TemporaryDirectory(prefix="faceblur_") as workspace:
            source = Path(workspace) / "input_video"
            source.write_bytes(payload)
            destination = Path(workspace) / output_name

            detector = build_detector(str(settings["detector"]), float(settings["confidence"]), report)
            stats = process_video(
                source,
                destination,
                detector,
                blur_mode=str(settings["blur_mode"]),
                strength=int(settings["strength"]),
                margin=float(settings["margin"]),
                feather=bool(settings["feather"]),
                smooth=bool(settings["smooth"]),
                stride=int(settings["stride"]),
                max_width=int(settings["max_width"]),
                crf=QUALITY_CRF[str(settings["quality"])],
                preset=QUALITY_PRESET[str(settings["quality"])],
                keep_audio=bool(settings["keep_audio"]),
                progress=report,
            )
            data = destination.read_bytes()

        progress_bar.progress(1.0, text="Done")
        status.update(label="Finished", state="complete", expanded=False)

        st.session_state["result"] = ProcessingResult(
            data=data,
            filename=output_name,
            stats=stats,
            settings=settings,
            source=(uploaded.name, len(payload)),
            kind=KIND_VIDEO,
        )
    except Exception as exc:  # noqa: BLE001 - surface the error to the user
        status.update(label="Processing failed", state="error")
        progress_bar.empty()
        st.error(f"{exc}")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    # The download widget and metrics are rendered on the next run; trigger it
    # so the result appears immediately below the uploader.
    st.rerun()


def _run_image(
    uploaded: st.runtime.uploaded_file_manager.UploadedFile,
    settings: dict[str, object],
    background: st.runtime.uploaded_file_manager.UploadedFile | None = None,
) -> None:
    payload = uploaded.getvalue()
    extension = Path(uploaded.name).suffix.lower().lstrip(".") or "png"
    stem = Path(uploaded.name).stem or "image"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_ ").strip() or "image"

    remove_background = bool(settings.get("remove_background", False))
    replace_background = remove_background and background is not None
    output_ext = _output_encoder(
        extension,
        remove_background=remove_background,
        replace_background=replace_background,
    )[0]
    suffix = "_replaced" if replace_background else "_nobg" if remove_background else "_blurred"
    output_name = f"{safe_stem}{suffix}{output_ext}"

    progress_bar = st.progress(0.0, text="Starting…")
    status = st.status("Processing image…", expanded=True)

    def report(fraction: float, message: str) -> None:
        progress_bar.progress(max(0.0, min(1.0, fraction)), text=message)
        status.update(label=message)

    try:
        detector = build_detector(str(settings["detector"]), float(settings["confidence"]), report)
        data, stats = process_image(
            payload,
            detector,
            blur_mode=str(settings["blur_mode"]),
            strength=int(settings["strength"]),
            margin=float(settings["margin"]),
            feather=bool(settings["feather"]),
            max_width=int(settings["max_width"]),
            source_extension=extension,
            remove_background=remove_background,
            background_image=background.getvalue() if background is not None else None,
            progress=report,
        )

        progress_bar.progress(1.0, text="Done")
        status.update(label="Finished", state="complete", expanded=False)

        st.session_state["result"] = ProcessingResult(
            data=data,
            filename=output_name,
            stats=stats,
            settings=settings,
            source=(
                uploaded.name,
                len(payload),
                (background.name, background.size) if background is not None else None,
            ),
            kind=KIND_IMAGE,
        )
    except Exception as exc:  # noqa: BLE001 - surface the error to the user
        status.update(label="Processing failed", state="error")
        progress_bar.empty()
        st.error(f"{exc}")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    # The preview and download widget are rendered on the next run.
    st.rerun()


def _run_morph(
    source: st.runtime.uploaded_file_manager.UploadedFile,
    target: st.runtime.uploaded_file_manager.UploadedFile,
    settings: dict[str, object],
) -> None:
    source_payload = source.getvalue()
    target_payload = target.getvalue()
    extension = Path(source.name).suffix.lower().lstrip(".") or "png"
    stem = Path(source.name).stem or "image"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_ ").strip() or "image"
    output_ext = _image_encoder(extension)[0]
    output_name = f"{safe_stem}_morph{output_ext}"

    progress_bar = st.progress(0.0, text="Starting…")
    status = st.status("Morphing faces…", expanded=True)

    def report(fraction: float, message: str) -> None:
        progress_bar.progress(max(0.0, min(1.0, fraction)), text=message)
        status.update(label=message)

    try:
        data, stats = process_morph_image(
            source_payload,
            target_payload,
            amount=float(settings["morph_amount"]),
            area=str(settings["morph_area"]),
            max_width=int(settings["max_width"]),
            source_extension=extension,
            finder=str(settings["morph_finder"]),
            progress=report,
        )

        progress_bar.progress(1.0, text="Done")
        status.update(label="Finished", state="complete", expanded=False)

        st.session_state["result"] = ProcessingResult(
            data=data,
            filename=output_name,
            stats=stats,
            settings=settings,
            source=(
                source.name,
                source.size,
                target.name,
                target.size,
            ),
            kind=KIND_MORPH,
        )
    except Exception as exc:  # noqa: BLE001 - surface the error to the user
        status.update(label="Processing failed", state="error")
        progress_bar.empty()
        st.error(f"{exc}")
        with st.expander("Technical details"):
            st.exception(exc)
        return

    # The preview and download widget are rendered on the next run.
    st.rerun()


def _forget_result(kind: str) -> None:
    result: ProcessingResult | None = st.session_state.get("result")
    if result is not None and result.kind == kind:
        st.session_state.pop("result", None)


def _show_result(source_key: tuple[object, ...], kind: str, settings: dict[str, object]) -> None:
    result: ProcessingResult | None = st.session_state.get("result")
    if result is None or result.kind != kind:
        return
    action = "Morph faces" if kind == KIND_MORPH else "Blur faces"
    if result.source != source_key:
        st.info(f"Input changed — click **{action}** to process the new file.")
        return
    if result.settings != settings:
        st.info(f"Settings changed — click **{action}** to re-process with the new options.")
    _render_result(result)


def _image_route(settings: dict[str, object]) -> None:
    uploaded = st.file_uploader(
        "Upload an image",
        type=list(IMAGE_EXTENSIONS),
        help="PNG, JPEG, WebP, BMP and TIFF are supported. Transparency is preserved when possible.",
    )

    background: st.runtime.uploaded_file_manager.UploadedFile | None = None
    if uploaded is not None and settings.get("remove_background"):
        background = st.file_uploader(
            "Background image (optional)",
            type=list(IMAGE_EXTENSIONS),
            key="background_image",
            help=(
                "Leave this empty to simply remove the background and keep the "
                "subject on a transparent canvas. If you add one, it becomes the "
                "output frame: the whole subject is resized to a natural size "
                "for that background instead of being cropped or stretched."
            ),
        )

    if uploaded is None:
        _forget_result(KIND_IMAGE)
        st.info("Upload an image to get started. Nothing leaves your computer.")
        with st.expander("How images are processed"):
            st.markdown(
                """
- Detection and blurring run locally with OpenCV/MediaPipe.
- The result keeps the original format, and PNG/WebP/TIFF transparency is preserved.
- **Remove background** in the sidebar segments the subject. Upload a background
  image above to replace it, or leave that empty for a transparent result.
  Segmentation is tuned for people, so portraits work best.
- A replacement background keeps its own aspect ratio and becomes the output
  frame. The whole subject is resized as one piece, centred, and given
  breathing room so detached parts and soft edges are never cropped.
- Use **Maximum width** in the sidebar to downscale very large photos.
                """
            )
        return

    source_key = (
        uploaded.name,
        uploaded.size,
        (background.name, background.size) if background is not None else None,
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(uploaded.getvalue(), caption=uploaded.name)
        st.caption(f"{uploaded.size / 1e6:.2f} MB · {uploaded.name}")

    with right:
        st.subheader("Options")
        summary = (
            f"**{settings['detector']}** · {settings['blur_mode']} at {settings['strength']}%"
        )
        if settings.get("remove_background"):
            summary += (
                " · background replaced (auto-fit)"
                if background is not None
                else " · background removed"
            )
        st.write(summary)
        if background is not None:
            st.image(background.getvalue(), caption=f"New background: {background.name}")
        start = st.button("Blur faces", type="primary", key="blur_image")

    if start:
        # Rendered outside the column so progress and errors span the page.
        _run_image(uploaded, settings, background)

    _show_result(source_key, KIND_IMAGE, settings)


def _morph_route(settings: dict[str, object]) -> None:
    left, right = st.columns(2)
    with left:
        source = st.file_uploader(
            "Source image (the base)",
            type=list(IMAGE_EXTENSIONS),
            key="morph_source",
            help="The result keeps this image's size, framing and background.",
        )
    with right:
        target = st.file_uploader(
            "Target image (the face to blend in)",
            type=list(IMAGE_EXTENSIONS),
            key="morph_target",
            help="The largest face in this image is aligned onto the source face.",
        )

    if source is None or target is None:
        _forget_result(KIND_MORPH)
        st.info(
            "Upload both a source and a target image to create a face morph. "
            "Nothing leaves your computer."
        )
        with st.expander("How face morphing works"):
            st.markdown(
                """
- The face in each photo is located with the chosen **finder** — MediaPipe
  FaceLandmarker in one pass, or YuNet first when the face is small in its
  photo — and a 478-point mesh is fitted. Models are downloaded once and
  cached locally.
- The face **ROI** is determined in both photos in a roll-normalised frame,
  so a tilted head reports the same ROI as an upright one. The whole target
  image is then resized with a single uniform, proportionate scale so its
  face ROI matches the source's in pixel size.
- The resized target face is placed **exactly** on the source face's
  footprint — same size, position and tilt, anchored on the eye midpoint —
  derived directly from the two ROIs rather than fitted to landmarks, so the
  morph stays proportionate at every amount.
- Both faces are warped to an in-between shape with a Delaunay-triangulated
  piecewise affine warp, then cross-dissolved into one **mixed face**. The
  target's skin tone is matched to the source so it does not look pasted.
- The mixed face is placed back on the **source image** inside a softly
  feathered face oval — the source background and framing stay untouched.
- **Morph amount** blends from the source (0%) to the target (100%).
- **Face only** keeps the source background. **Whole image** also blends the
  aligned target background, which works best with similar compositions.
- Use clear, front-facing photos where the face is reasonably large.
                """
            )
        return

    source_key = (source.name, source.size, target.name, target.size)
    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.subheader("Source")
        st.image(source.getvalue(), caption=source.name)
        st.caption(f"{source.size / 1e6:.2f} MB · {source.name}")
    with preview_right:
        st.subheader("Target")
        st.image(target.getvalue(), caption=target.name)
        st.caption(f"{target.size / 1e6:.2f} MB · {target.name}")

    options_left, options_right = st.columns([3, 1])
    with options_left:
        st.write(
            f"**Morph amount:** {round(float(settings['morph_amount']) * 100)}% · "
            f"**Blend area:** {settings['morph_area']} · "
            f"**Finder:** {settings['morph_finder']}"
        )
    with options_right:
        start = st.button("Morph faces", type="primary", key="morph_faces")

    if start:
        # Rendered outside the column so progress and errors span the page.
        _run_morph(source, target, settings)

    _show_result(source_key, KIND_MORPH, settings)


def _video_route(settings: dict[str, object]) -> None:
    uploaded = st.file_uploader(
        "Upload a video",
        type=list(VIDEO_EXTENSIONS),
        help="MP4/MOV work best. Other containers are converted to MP4.",
    )

    if uploaded is None:
        _forget_result(KIND_VIDEO)
        st.info("Upload a video to get started. Nothing leaves your computer.")
        with st.expander("What changed compared to the old app?"):
            st.markdown(
                """
- `mp.solutions.face_detection` no longer exists (removed in MediaPipe 0.10.31),
  so detection now uses the MediaPipe Tasks API or OpenCV's YuNet.
- Models are downloaded once and cached, because MediaPipe stopped bundling them.
- Output is H.264/yuv420p instead of `mp4v`, so it plays in every browser.
- The original audio track is copied back into the result.
- Files are handled in a temp directory instead of the working directory, and
  the download no longer deletes the file or re-runs the whole pipeline.
                """
            )
        return

    source_key = (uploaded.name, uploaded.size)
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        if uploaded.name.lower().endswith((".mp4", ".mov", ".m4v", ".webm")):
            uploaded.seek(0)
            st.video(uploaded)
        else:
            st.info(f"{uploaded.name} cannot be previewed in the browser, but it can be processed.")
        st.caption(f"{uploaded.size / 1e6:.1f} MB")

    with right:
        st.subheader("Options")
        st.write(
            f"**{settings['detector']}** · {settings['blur_mode']} at "
            f"{settings['strength']}% · detect every {settings['stride']} frame(s)"
        )
        start = st.button("Blur faces", type="primary", key="blur_video")

    if start:
        # Rendered outside the column so progress and errors span the page.
        _run_video(uploaded, settings)

    _show_result(source_key, KIND_VIDEO, settings)


def main() -> None:
    if not _check_runtime():
        st.stop()

    _page_config()

    st.title("🎭 FaceBlur Studio")
    st.caption(
        "Blur or pixelate every face in an image or a video, entirely on your machine. "
        "Images can also have their background removed or replaced, and two faces "
        "can be morphed into one. Built with OpenCV 5, MediaPipe Tasks and Streamlit."
    )

    kind = st.radio(
        "What would you like to do?",
        (KIND_IMAGE, KIND_VIDEO, KIND_MORPH),
        index=None,
        horizontal=True,
        help="Pick a type first; the uploader and options below adapt to it.",
    )
    if kind is None:
        st.info(
            "Choose **Image**, **Video** or **Face morph** above to get started. "
            "Nothing leaves your computer."
        )
        st.caption(
            "Images are blurred and re-encoded in memory, with optional background "
            "removal or replacement. Videos are re-encoded to H.264 with the original "
            "audio copied back in. Face morph blends a source and a target photo."
        )
        return

    st.caption(
        "Image mode: blur, then optionally remove the background or replace it with "
        "your own image."
        if kind == KIND_IMAGE
        else (
            "Video mode: the result is H.264 with the original audio copied back in."
            if kind == KIND_VIDEO
            else "Face morph mode: upload a source and a target image, then blend the "
            "largest face from each."
        )
    )

    settings = _sidebar(str(kind))

    if kind == KIND_IMAGE:
        _image_route(settings)
    elif kind == KIND_VIDEO:
        _video_route(settings)
    else:
        _morph_route(settings)


if __name__ == "__main__":
    main()
