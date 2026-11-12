"""FaceBlur Studio — image/video face blurring, background editing and morphing.

This is ``v10.py``: image/video face blurring plus face morphing (photos and
videos). The app first asks whether the user is uploading an image, a video,
images to morph, or a video to morph, then routes to the matching flow. The image
and video flows share the detector, blur and settings code: every face
found in the frame gets the same blur treatment. In image mode the
background can optionally be removed (transparent result) or replaced with
a user-supplied background image: the background keeps its native aspect
ratio and the subject is scaled and placed to match it.

What the user sees was reworked so previews always match the uploaded
files: the source image is shown exactly as uploaded, detected faces are
no longer outlined on it — each face is identified by the numbered
thumbnail beside its own target uploader instead — and every target
picture is previewed immediately once it is uploaded. All processing still
runs in OpenCV's BGR channel order internally; only what is sent to the
screen is converted to RGB, so a preview never differs from the upload.

The face-morph flow detects **all** faces in the source image and asks for a
separate target picture for each of them; faces without a target are left
completely untouched. For every paired face, the face ROI is determined in a
roll-normalised frame (a tilted head reports the same ROI as an upright
one). The whole target image is resized with a single uniform scale so its
face ROI matches the source face's in pixel size, then the target face is
placed *exactly* on the source face's footprint — same size, position and
tilt, anchored on the eye midpoint — so the mixed face replaces the source
face proportionately instead of being overlaid at a compromise position.
Both faces are warped to an intermediate shape with a Delaunay-triangulated
piecewise warp. The two warped faces are cross-dissolved into a single mixed
face — with the target's skin tone matched to the source — and placed back
on the source image inside a feathered face oval, so the source framing,
background and all unpaired faces are preserved. The faces can be located
by MediaPipe's FaceLandmarker in one pass — with a YuNet sweep adding any
faces its own detector misses — or by YuNet first when a face is small in
its photo, and the morph amount is adjustable from 0% (source) to 100%
(target). The blend can cover the face only or the whole image.

The video-morph flow first scans evenly spread frames of the upload to
learn which **distinct faces** appear in it — every face is followed by
position and timing as it moves, appears and leaves — and shows one
numbered thumbnail per tracked face, each with its own optional target
picture. Processing then re-runs the image-morph pipeline on every frame:
each detected face is matched to its tracked face by position, morphed
toward that face's target inside its feathered oval, and everything else
in the frame is preserved. The result is an H.264 MP4 with the original
audio copied back in. Faces without a target, faces the matcher cannot
follow, and people who appear only briefly stay untouched.

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
    streamlit run v10.py
"""

from __future__ import annotations

import math
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
KIND_VIDEO_MORPH = "Video morph"

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

# Upper bound on faces handled per source image. The morph UI shows one
# target uploader per detected face, so this also caps the widget count.
MAX_MORPH_FACES = 8

# Maximum size for any single image upload (20 MB). Videos are exempt from
# this and may be up to server.maxUploadSize (2 GB). Images larger than this
# are rejected before they consume memory.
IMAGE_MAX_SIZE = 20 * 1024 * 1024  # 20 MB

# Video morphing: how many evenly spread frames are scanned to learn which
# distinct faces appear in a video, and how far — in multiples of the larger
# face's size — a detected face may sit from a tracked face's last known
# position to still count as the same person.
VIDEO_SCAN_SAMPLES = 36
SLOT_MATCH_RATIO = 1.25

# Maximum combined cost (appearance + position) for matching one face to a
# track. Lower = stricter; only the same person across pose changes should
# clear it.
MATCH_THRESHOLD = 0.50

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
    """An axis-aligned face box in pixel coordinates.

    ``identity`` is a stable re-identification id assigned by
    :class:`FaceReIdTracker`. ``-1`` means the box has not been matched to a
    known person yet.
    """

    x: int
    y: int
    w: int
    h: int
    score: float = 0.0
    identity: int = -1


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


@dataclass
class VideoFaceSlot:
    """One distinct face recognised while scanning a video.

    ``points``/``preview`` come from the clearest (largest) sighting and are
    what the UI numbers for the user; ``center_x``/``center_y``/``size``
    freeze where the face was *first* seen, which is where the processing
    pass starts following it, and ``first_frame``/``last_frame`` bound when
    it was observed.
    """

    points: np.ndarray
    preview: np.ndarray
    center_x: float
    center_y: float
    size: float
    first_frame: int
    last_frame: int
    sightings: int


@dataclass
class VideoFaceScan:
    """Everything the video-morph pipeline needs to know about a video."""

    slots: list[VideoFaceSlot]
    samples: int  # frames actually scanned
    frames: int  # total frames reported by the container (0 if unknown)
    step: int  # frames between scanned samples (the tracker's time slack)
    width: int
    height: int
    fps: float


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
# Face re-identification
# --------------------------------------------------------------------------- #


# Maximum number of appearance exemplars kept per identity. A gallery lets one
# person be recognised across pose changes (frontal, three-quarter, profile)
# because each view stores its own descriptor and a detection matches against
# the closest view instead of a single averaged one.
GALLERY_SIZE = 6

# Confirmation: a new identity must be seen this many frames before it counts
# as a real tracked face. Prevents one person being split into several
# short-lived identities while the detector wobbles.
CONFIRM_HITS = 3

# An identity is forgotten only after this many consecutive missed frames.
MAX_LOST = 90

# When a track has been lost this long, a re-detection must match by appearance
# alone (position is no longer trusted) to reclaim it.
REAPPEAR_FRAMES = 30


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two descriptors (0 = identical, 2 = opposite)."""
    if a is None or b is None:
        return 2.0
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 2.0
    return float(1.0 - float(np.dot(a, b)) / (na * nb))


def _landmark_descriptor(points: np.ndarray) -> np.ndarray | None:
    """Roll- and scale-normalised face descriptor from MediaPipe landmarks.

    Landmarks are recentered on the eye midpoint, rotated so the eye line is
    horizontal and scaled by the eye distance, then L2-normalised. Translation,
    scale and roll invariant, so the same person yields a similar vector across
    small pose changes.
    """
    if points is None or len(points) < max(FACE_OVAL_INDICES) + 1:
        return None
    if len(points) > LEFT_IRIS_INDEX:
        right_eye = points[RIGHT_IRIS_INDEX]
        left_eye = points[LEFT_IRIS_INDEX]
    else:
        corners = points[list(EYE_CORNER_INDICES)]
        right_eye = (corners[0] + corners[1]) / 2.0
        left_eye = (corners[2] + corners[3]) / 2.0
    anchor = (right_eye + left_eye) / 2.0
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist < 1e-3:
        extent = points.max(axis=0) - points.min(axis=0)
        eye_dist = float(np.linalg.norm(extent))
    if eye_dist < 1e-3:
        return None
    roll = float(np.arctan2(left_eye[1] - right_eye[1],
                            left_eye[0] - right_eye[0]))
    cos_r, sin_r = float(np.cos(-roll)), float(np.sin(-roll))
    shifted = points - anchor
    rotated = np.stack(
        [shifted[:, 0] * cos_r - shifted[:, 1] * sin_r,
         shifted[:, 0] * sin_r + shifted[:, 1] * cos_r],
        axis=1,
    )
    oval = rotated[list(FACE_OVAL_INDICES)]
    desc = (oval / eye_dist).ravel()
    norm_v = float(np.linalg.norm(desc))
    if norm_v < 1e-6:
        return None
    return (desc / norm_v).astype(np.float32)


def _appearance_descriptor(image_bgr: np.ndarray, box: FaceBox) -> np.ndarray | None:
    """Robust appearance signature from a face crop (box-only fallback).

    Combines an HSV colour histogram (skin/clothing tone) with a gradient
    orientation histogram (face shape / hairline), both computed on a fixed
    grid and L2-normalised. Much more stable than raw pixels across pose and
    lighting changes, yet needs no landmark model.
    """
    x0, y0 = max(0, box.x), max(0, box.y)
    x1 = min(image_bgr.shape[1], box.x + box.w)
    y1 = min(image_bgr.shape[0], box.y + box.h)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    small = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)

    parts: list[np.ndarray] = []

    # HSV colour histogram (8 bins per channel), L1-normalised per channel.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [8], [0, 256])
        hist = hist.ravel()
        s = float(hist.sum())
        if s > 1e-9:
            hist = hist / s
        parts.append(hist)

    # Gradient orientation histogram (9 bins) on the luminance channel.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # [0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    hist, _ = np.histogram(ang.ravel(), bins=9, range=(0.0, 1.0), weights=mag.ravel())
    s = float(hist.sum())
    if s > 1e-9:
        hist = hist / s
    parts.append(hist.astype(np.float32))

    desc = np.concatenate(parts).astype(np.float32)
    n = float(np.linalg.norm(desc))
    if n < 1e-9:
        return None
    return (desc / n).astype(np.float32)


@dataclass
class FaceIdentity:
    """One tracked person with a gallery of appearance exemplars.

    ``gallery`` holds up to ``GALLERY_SIZE`` descriptors captured at different
    poses/expressions, so a detection matches against the closest view rather
    than a single averaged vector. ``last_box``/``velocity`` predict where the
    person should appear next (constant-velocity motion model), which keeps the
    track alive through brief angle or position changes.
    """

    identity: int
    gallery: list[np.ndarray]
    last_box: FaceBox | None
    velocity: tuple[float, float]
    last_seen: int
    lost: int
    hits: int
    confirmed: bool
    color: tuple[int, int, int]


class FaceReIdTracker:
    """Assigns a stable id to every face, local and free of charge.

    Each identity keeps a small gallery of appearance exemplars plus a
    constant-velocity motion prediction. A detection is matched to the
    identity whose closest exemplar is nearest (appearance) and whose predicted
    position is nearest (motion). Identities survive being lost for many
    frames and can be reclaimed by appearance alone, so one person turning
    from frontal to profile — or briefly leaving the frame — is tracked as a
    single face instead of several.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        max_lost: int = MAX_LOST,
        position_gate: float = 2.0,
        appearance_weight: float = 0.55,
    ) -> None:
        self.threshold = threshold
        self.max_lost = max_lost
        self.position_gate = position_gate
        self.appearance_weight = appearance_weight
        self.identities: list[FaceIdentity] = []
        self.next_id = 0

    def _colour(self, identity: int) -> tuple[int, int, int]:
        h = identity * 0.61803398875
        return (
            int(255 * (0.55 + 0.45 * np.sin(h))),
            int(255 * (0.55 + 0.45 * np.sin(h + 2.1))),
            int(255 * (0.55 + 0.45 * np.sin(h + 4.2))),
        )

    @staticmethod
    def _centre(box: FaceBox) -> tuple[float, float]:
        return (box.x + box.w / 2.0, box.y + box.h / 2.0)

    def _predicted_box(self, ident: FaceIdentity) -> FaceBox | None:
        """Where the identity should be now, from its last box + velocity."""
        if ident.last_box is None:
            return None
        last = ident.last_box
        vx, vy = ident.velocity
        return FaceBox(
            x=round(last.x + vx), y=round(last.y + vy), w=last.w, h=last.h,
            score=last.score, identity=ident.identity,
        )

    def _position_cost(self, box: FaceBox, ident: FaceIdentity) -> float:
        predicted = self._predicted_box(ident)
        if predicted is None:
            return 0.0
        cx, cy = self._centre(box)
        px, py = self._centre(predicted)
        reach = self.position_gate * max(box.w, box.h, predicted.w, predicted.h)
        if reach < 1e-3:
            return 0.0
        return float(math.hypot(cx - px, cy - py) / reach)

    def _appearance_cost(self, desc: np.ndarray | None, ident: FaceIdentity) -> float:
        if desc is None or not ident.gallery:
            return 0.5  # neutral when appearance is unavailable
        return float(min(_cosine_distance(desc, g) for g in ident.gallery))

    def _signature(
        self,
        image_bgr: np.ndarray,
        box: FaceBox,
        points: np.ndarray | None,
    ) -> np.ndarray | None:
        if points is not None and len(points) >= 5:
            return _landmark_descriptor(points)
        return _appearance_descriptor(image_bgr, box)

    def _add_to_gallery(self, ident: FaceIdentity, desc: np.ndarray | None) -> None:
        if desc is None:
            return
        for existing in ident.gallery:
            if _cosine_distance(desc, existing) < 0.15:
                return  # near-duplicate of an already-stored view
        ident.gallery.append(desc.astype(np.float32))
        if len(ident.gallery) > GALLERY_SIZE:
            ident.gallery.pop(0)

    def update(
        self,
        image_bgr: np.ndarray,
        boxes: Sequence[FaceBox],
        points_by_index: dict[int, np.ndarray] | None = None,
        frame_index: int = 0,
    ) -> list[FaceBox]:
        """Match/register ``boxes`` to identities, returning them with ids set.

        Boxes are processed strongest-first. Each detection is matched to the
        identity minimising ``appearance_weight * appearance_cost +
        (1 - appearance_weight) * position_cost``. A match is accepted when
        both terms are individually plausible OR the combined cost is below
        threshold — this lets a face turning sideways (appearance drifts, but
        position stays close) keep its id, while a stranger standing where
        someone left (position close, appearance different) does not steal it.

        New identities begin unconfirmed and only become real tracked faces
        after ``CONFIRM_HITS`` sightings, which stops one person being split
        into several short-lived identities while the detector wobbles.
        """
        points_by_index = points_by_index or {}
        detections: list[tuple[int, FaceBox, np.ndarray | None]] = []
        for index, box in enumerate(boxes):
            desc = self._signature(image_bgr, box, points_by_index.get(index))
            detections.append((index, box, desc))

        detections.sort(
            key=lambda item: (item[1].score, item[1].w * item[1].h), reverse=True
        )

        used_identities: set[int] = set()
        updated: dict[int, FaceBox] = {}

        for original_index, box, desc in detections:
            best_ident: int | None = None
            best_cost = float("inf")
            for ident in self.identities:
                if ident.identity in used_identities:
                    continue
                pos_cost = self._position_cost(box, ident)
                # Hard gate: too far from predicted position rules out this track.
                if pos_cost > 1.0:
                    continue
                app_cost = self._appearance_cost(desc, ident)
                combined = (
                    self.appearance_weight * app_cost
                    + (1.0 - self.appearance_weight) * pos_cost
                )
                # Accept if combined cost is low, or if position is very close
                # and appearance isn't actively contradictory.
                position_ok = pos_cost < 0.35
                appearance_ok = app_cost < self.threshold
                if combined < best_cost and (combined < self.threshold
                                             or (position_ok and appearance_ok)):
                    best_cost = combined
                    best_ident = ident.identity

            if best_ident is not None:
                ident = next(i for i in self.identities if i.identity == best_ident)
                used_identities.add(best_ident)
                ident.lost = 0
                ident.hits += 1
                if ident.hits >= CONFIRM_HITS:
                    ident.confirmed = True

                # Motion prediction: constant velocity from last box to this box.
                if ident.last_box is not None:
                    cx, cy = self._centre(box)
                    lx, ly = self._centre(ident.last_box)
                    vx, vy = ident.velocity
                    # EMA on velocity so it adapts smoothly.
                    ident.velocity = (
                        0.7 * vx + 0.3 * (cx - lx),
                        0.7 * vy + 0.3 * (cy - ly),
                    )
                else:
                    ident.velocity = (0.0, 0.0)

                # Smoothed box.
                if ident.last_box is not None:
                    a = 0.45
                    smoothed = FaceBox(
                        x=round(ident.last_box.x + a * (box.x - ident.last_box.x)),
                        y=round(ident.last_box.y + a * (box.y - ident.last_box.y)),
                        w=round(ident.last_box.w + a * (box.w - ident.last_box.w)),
                        h=round(ident.last_box.h + a * (box.h - ident.last_box.h)),
                        score=box.score,
                        identity=best_ident,
                    )
                else:
                    smoothed = box
                ident.last_box = FaceBox(
                    x=smoothed.x, y=smoothed.y, w=smoothed.w, h=smoothed.h,
                    score=smoothed.score, identity=best_ident,
                )
                self._add_to_gallery(ident, desc)
                updated[original_index] = ident.last_box
            else:
                new_id = self.next_id
                self.next_id += 1
                self.identities.append(
                    FaceIdentity(
                        identity=new_id,
                        gallery=[desc] if desc is not None else [],
                        last_box=box,
                        velocity=(0.0, 0.0),
                        last_seen=frame_index,
                        lost=0,
                        hits=1,
                        confirmed=False,
                        color=self._colour(new_id),
                    )
                )
                updated[original_index] = FaceBox(
                    x=box.x, y=box.y, w=box.w, h=box.h,
                    score=box.score, identity=new_id,
                )

        # Age unmatched identities; forget only after max_lost consecutive misses.
        remaining: list[FaceIdentity] = []
        for ident in self.identities:
            if ident.identity in used_identities:
                remaining.append(ident)
                continue
            ident.lost += 1
            if ident.lost <= self.max_lost:
                remaining.append(ident)
        self.identities = remaining

        out: list[FaceBox] = []
        for i in range(len(boxes)):
            out.append(updated.get(i, boxes[i]))
        return out

    def confirmed_identities(self) -> int:
        """Number of identities confirmed as real tracked faces."""
        return sum(1 for ident in self.identities if ident.confirmed)


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


def _output_frame_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Even, encoder-friendly frame size after the optional width cap.

    Shared by every video path — blur, the morph scan and the morph itself —
    so the meshes a scan produced always line up with the frames the
    processing pass works on.
    """
    scale = 1.0
    if max_width and width > max_width:
        scale = max_width / width
    return _even(width * scale), _even(height * scale)


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
        out_width, out_height = _output_frame_size(source_width, source_height, max_width)

        sink = make_sink(
            destination,
            out_width,
            out_height,
            fps,
            crf=crf,
            preset=preset,
            audio_source=source if keep_audio else None,
        )

        tracker = FaceReIdTracker()
        boxes: list[FaceBox] = []
        confirmed_ids: set[int] = set()
        frame_index = 0
        stride = max(1, int(stride))

        while True:
            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(
                    frame, (out_width, out_height), interpolation=cv2.INTER_AREA
                )

            if frame_index % stride == 0:
                detected = detector.detect(frame, int(frame_index * 1000 / fps))
                boxes = tracker.update(frame, detected, None, frame_index)
                stats.detection_frames += 1
                for box in boxes:
                    if box.identity >= 0:
                        confirmed_ids.add(box.identity)
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
        stats.faces = len(confirmed_ids)
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
    """MediaPipe Tasks FaceLandmarker, used to find faces and their meshes.

    The task returns a 478-point mesh (468 face points plus iris points) for
    every face it finds. ``all_faces`` returns one mesh per face — largest
    first, capped at ``MAX_MORPH_FACES`` — which drives the multi-face morph
    flow; ``largest_face`` returns just the biggest mesh, which is what a
    morph *target* image uses. Two finders can steer where the meshes are
    fitted:

    - ``MORPH_FINDER_MEDIAPIPE`` (default): the landmarker's own detector
      finds the faces in one pass and is the most accurate when each face
      fills a fair share of the frame. Its detector gives up on faces that
      are small relative to the photo, so a YuNet sweep adds any faces it
      missed to the result.
    - ``MORPH_FINDER_YUNET``: YuNet locates the face boxes first and each
      mesh is fitted on a crop normalised around its box, which keeps
      working when faces are small relative to the photo.

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
                num_faces=MAX_MORPH_FACES,
            )
        )
        # Lazily created and reused, so per-frame sweeps (video morphing)
        # do not reload the YuNet model for every single frame.
        self._yunet: "YunetFaceDetector | None" = None

    def largest_face(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        """Return pixel-space landmarks for the largest face in ``image_bgr``."""
        return self.all_faces(image_bgr, label)[0]

    def all_faces(
        self, image_bgr: np.ndarray, label: str = "image"
    ) -> list[np.ndarray]:
        """Return pixel-space landmarks for every face, largest first.

        At most ``MAX_MORPH_FACES`` meshes are returned. Raises
        ``RuntimeError`` with a user-facing message when neither finder can
        locate a face.
        """
        if self.finder == MORPH_FINDER_YUNET:
            meshes = self._landmark_all_via_yunet(image_bgr)
            if meshes:
                return meshes
            meshes = self._detect_all(image_bgr)
            if meshes:
                return meshes
            raise RuntimeError(
                f"No face was detected in the {label} image. Use a clear, "
                "front-facing photo where the face is reasonably large."
            )

        # The landmarker's own detector gives up on faces that are small
        # relative to the photo (and on very wide, crowded compositions), so
        # YuNet sweeps for any faces it missed and the genuinely new ones —
        # no overlap with an already-found face — are merged in.
        meshes = self._merge_disjoint(
            self._detect_all(image_bgr), self._landmark_all_via_yunet(image_bgr)
        )
        if meshes:
            return meshes
        raise RuntimeError(
            f"No face was detected in the {label} image. Use a clear, "
            "front-facing photo where the face is reasonably large."
        )

    def find_faces(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Like :meth:`all_faces`, but a frame with no face is not an error.

        Video frames legitimately contain no face — somebody walked out of
        the shot, the camera panned away — so the video-morph pipeline calls
        this instead of :meth:`all_faces` and simply leaves such frames
        untouched. Only the "no face found" message is treated as empty; any
        other failure still propagates.
        """
        try:
            return self.all_faces(image_bgr, "frame")
        except RuntimeError:
            return []

    @staticmethod
    def _mesh_area(points: np.ndarray) -> float:
        """Bounding-box area of a mesh's face oval."""
        oval = points[list(FACE_OVAL_INDICES)]
        extents = oval.max(axis=0) - oval.min(axis=0)
        return float(extents[0] * extents[1])

    @staticmethod
    def _merge_disjoint(
        meshes: list[np.ndarray], extra: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Append meshes from ``extra`` whose face does not overlap ``meshes``.

        Two detectors agreeing on the same face produce heavily overlapping
        oval bounding boxes; a candidate is only kept when its oval overlaps
        no face already in the list, so the merge adds faces one detector
        missed without ever duplicating a face. The result is sorted largest
        first and capped at ``MAX_MORPH_FACES``.
        """

        def bounds(points: np.ndarray) -> tuple[float, float, float, float]:
            oval = points[list(FACE_OVAL_INDICES)]
            x_min, y_min = oval.min(axis=0)
            x_max, y_max = oval.max(axis=0)
            return float(x_min), float(y_min), float(x_max), float(y_max)

        def overlaps(
            a: tuple[float, float, float, float], b: tuple[float, float, float, float]
        ) -> bool:
            inter_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            inter_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = inter_x * inter_y
            if inter <= 0.0:
                return False
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            return union > 0.0 and inter / union > 0.3

        merged = list(meshes)
        known = [bounds(points) for points in merged]
        for candidate in extra:
            candidate_bounds = bounds(candidate)
            if not any(overlaps(candidate_bounds, box) for box in known):
                merged.append(candidate)
                known.append(candidate_bounds)
        merged.sort(key=FaceLandmarkerModel._mesh_area, reverse=True)
        return merged[:MAX_MORPH_FACES]

    def _detect_all(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Run the landmarker directly: one mesh per face, largest first."""
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._landmarker.detect(mp_image)
        meshes = [
            np.array(
                [[point.x * width, point.y * height] for point in face], np.float32
            )
            for face in result.face_landmarks[:MAX_MORPH_FACES]
        ]
        meshes = [points for points in meshes if len(points) > max(FACE_OVAL_INDICES)]
        meshes.sort(key=self._mesh_area, reverse=True)
        return meshes

    def _detect_largest(self, image_bgr: np.ndarray, label: str) -> np.ndarray:
        """Largest mesh only; raises when nothing is found."""
        meshes = self._detect_all(image_bgr)
        if not meshes:
            raise RuntimeError(
                f"No face was detected in the {label} image. Use a clear, "
                "front-facing photo where the face is reasonably large."
            )
        return meshes[0]

    def _landmark_all_via_yunet(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Locate faces with YuNet, then fit a mesh on each normalised crop.

        YuNet is a multi-scale sliding-window detector, so it finds face
        boxes even when the faces are small relative to the photo — exactly
        where the landmarker's own detector gives up. Faces come back largest
        first, capped at ``MAX_MORPH_FACES``; boxes whose mesh cannot be
        fitted are skipped.
        """
        try:
            if self._yunet is None:
                model_path = ensure_model(YUNET_FILENAME, YUNET_URLS, YUNET_MIN_BYTES)
                self._yunet = YunetFaceDetector(
                    model_path, score_threshold=0.5, max_side=1920
                )
            boxes = self._yunet.detect(image_bgr)
        except Exception:  # noqa: BLE001 - the caller keeps the original error
            return []
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda box: box.w * box.h, reverse=True)
        meshes: list[np.ndarray] = []
        for box in boxes[:MAX_MORPH_FACES]:
            points = self._landmark_around_box(image_bgr, box)
            if points is not None:
                meshes.append(points)
        return meshes

    def _landmark_around_box(
        self, image_bgr: np.ndarray, box: FaceBox
    ) -> np.ndarray | None:
        """Fit the mesh on a crop normalised around one detected face box.

        The crop carries enough context that the face fills roughly a quarter
        of it — squarely inside the landmarker's detection sweet spot — and
        it is rescaled to a working size before landmarking. The points are
        mapped back into the original pixel space. Returns ``None`` when the
        mesh cannot be fitted on the crop.
        """
        height, width = image_bgr.shape[:2]
        center_x = box.x + box.w / 2.0
        center_y = box.y + box.h / 2.0
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
            points = self._detect_largest(resized, "crop")
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
        if self._yunet is not None:
            try:
                self._yunet.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
            self._yunet = None


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


def _morph_face_pair(
    working: np.ndarray,
    source_points: np.ndarray,
    target: np.ndarray,
    target_points: np.ndarray,
    amount: float,
    whole_image: bool,
) -> tuple[np.ndarray, float]:
    """Morph one source face toward one target face, compositing on ``working``.

    ``working`` is the current result image; the morphed face is blended into
    it and the new image is returned, so several faces can be morphed one
    after another (later faces see earlier morphs, which keeps overlapping
    faces consistent). Also returns the proportionate resize factor that was
    applied to the target, for the result summary.
    """
    source_shape = working.shape
    # Roll-normalised ROI frame of each face. The proportionate resize factor
    # and the exact placement transform both derive from these.
    resize_factor = _roi_scale_ratio(
        _face_frame(source_points), _face_frame(target_points)
    )

    # Bring the target to the source face's resolution before any warping
    # happens, so neither face is starved or blown up.
    target, target_points = _match_target_to_source_roi(
        target, target_points, resize_factor, source_shape
    )

    # Place the resized target face exactly on the source face's footprint.
    aligned_target, aligned_points = _align_target(
        target, target_points, source_points, source_shape
    )
    morph_points = (
        (1.0 - amount) * source_points + amount * aligned_points
    ).astype(np.float32)

    if whole_image:
        boundary = _image_boundary_points(source_shape)
        source_all = np.vstack([source_points, boundary])
        target_all = np.vstack([aligned_points, boundary])
        morph_all = np.vstack([morph_points, boundary])
    else:
        source_all = np.vstack([source_points, _face_ring_points(source_points)])
        target_all = np.vstack([aligned_points, _face_ring_points(aligned_points)])
        morph_all = np.vstack([morph_points, _face_ring_points(morph_points)])
    triangles = _triangulate(morph_all)
    warped_source = _warp_piecewise(working, source_all, morph_all, triangles)
    warped_target = _warp_piecewise(
        aligned_target, target_all, morph_all, triangles
    )

    # Match the target's tone to the source inside the face region so the
    # mixed face looks native to the source photo.
    warped_target = _match_face_colors(warped_source, warped_target, morph_points)

    if whole_image:
        blended = (
            warped_source.astype(np.float32) * (1.0 - amount)
            + warped_target.astype(np.float32) * amount
        )
    else:
        # Cross-dissolve the two warped faces into the mixed face. The
        # target's share fades out with the mask, so none of the target's
        # background leaks in around the face; then the mixed face is placed
        # on the working image inside the feathered oval, leaving everything
        # outside this face — background, framing and other faces — untouched.
        mask = _face_blend_mask(source_shape, morph_points)
        target_weight = mask * amount
        mixed = (
            warped_source.astype(np.float32) * (1.0 - target_weight)[:, :, None]
            + warped_target.astype(np.float32) * target_weight[:, :, None]
        )
        blended = (
            working.astype(np.float32) * (1.0 - mask)[:, :, None]
            + mixed * mask[:, :, None]
        )
    return np.clip(blended, 0, 255).astype(np.uint8), resize_factor


def process_morph_image(
    source_payload: bytes,
    target_payloads: Sequence[bytes | None] | bytes,
    *,
    amount: float,
    area: str,
    max_width: int,
    source_extension: str,
    finder: str = MORPH_FINDER_MEDIAPIPE,
    progress: ProgressCallback | None = None,
) -> tuple[bytes, ProcessStats]:
    """Morph every paired face of the source image toward its target face.

    ``target_payloads`` holds one entry per face detected in the source
    image, largest face first; a ``None`` entry means that source face is
    left completely untouched. (A single ``bytes`` value is accepted as
    shorthand for the largest face only.) ``amount`` runs from 0 (source) to
    1 (target) and is shared by every morphed face. The pipeline per paired
    face is:

    1. Determine the ROI of the source face and of the target's largest
       face in roll-normalised frames (a tilted head reports the same ROI
       as an upright one).
    2. Resize the whole target image with a single uniform, proportionate
       scale so its face ROI matches the source face ROI in pixel size.
    3. Place the resized target face *exactly* on the source face's
       footprint — same size, position and tilt, anchored on the eye
       midpoint — derived directly from the two ROIs, not fitted.
    4. Warp both faces to the intermediate face shape with a Delaunay-
       triangulated piecewise-affine warp and cross-dissolve them, with the
       target's skin tone matched to the source.
    5. Place that mixed face back on the working image inside a feathered
       face oval, so the source background, framing and all unpaired faces
       are preserved.

    ``area`` limits the blend to each face (default). Whole-image blending
    is only meaningful for a single morphed face — with several faces every
    morph stays inside its own oval so the faces do not overwrite each
    other. ``finder`` picks how the faces are located:
    ``MORPH_FINDER_MEDIAPIPE`` (one pass) or ``MORPH_FINDER_YUNET`` (YuNet
    locates the boxes first, which is more reliable for small faces).
    """
    if isinstance(target_payloads, (bytes, bytearray)):
        target_list: list[bytes | None] = [bytes(target_payloads)]
    else:
        target_list = [
            bytes(payload) if payload is not None else None
            for payload in target_payloads
        ]

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
            progress(0.15, "Decoding the source image…")
        source = _decode_image_bgr(source_payload, "source")
        if max_width:
            source = _limit_width(source, max_width)

        if progress is not None:
            progress(0.3, "Detecting faces in the source…")
        source_faces = landmarker.all_faces(source, "source")

        # Pair each detected face with its optional target; missing entries
        # are treated as "leave this face alone".
        if len(target_list) < len(source_faces):
            target_list += [None] * (len(source_faces) - len(target_list))
        pairs = [
            (index, points, payload)
            for index, (points, payload) in enumerate(
                zip(source_faces, target_list)
            )
            if payload is not None
        ]

        # Whole-image blending replaces the entire frame with the aligned
        # target, which only makes sense for a single morphed face; with
        # several faces every morph stays inside its own oval so the faces
        # do not overwrite each other.
        whole_image = area == MORPH_WHOLE_IMAGE and len(pairs) == 1
        area_note = (
            " · whole image applies to a single face; blended per face oval"
            if area == MORPH_WHOLE_IMAGE and pairs and not whole_image
            else ""
        )

        result = source.copy()
        factors: list[tuple[int, float]] = []
        if amount > 1e-4:
            for step, (index, points, payload) in enumerate(pairs):
                if progress is not None:
                    progress(
                        0.4 + 0.5 * step / max(1, len(pairs)),
                        f"Morphing face {index + 1} of {len(source_faces)}…",
                    )
                try:
                    target = _decode_image_bgr(payload, f"target for face {index + 1}")
                    if max_width:
                        target = _limit_width(target, max_width)
                    target_points = landmarker.largest_face(
                        target, f"target for face {index + 1}"
                    )
                except RuntimeError as exc:
                    raise RuntimeError(f"Face {index + 1}: {exc}") from exc
                result, factor = _morph_face_pair(
                    result, points, target, target_points, amount, whole_image
                )
                factors.append((index + 1, factor))

        if progress is not None:
            progress(0.92, "Encoding result…")
        encode_ext, encoder_label, params = _image_encoder(source_extension)
        ok, buffer = cv2.imencode(encode_ext, result, params)
        if not ok:
            raise RuntimeError(f"Could not encode the result as {encoder_label}.")

        stats.faces = len(factors)
        summary = (
            f"{len(factors)} of {len(source_faces)} source "
            f"face{'s' if len(source_faces) != 1 else ''} morphed"
        )
        if factors:
            summary += " (" + ", ".join(
                f"face {number}: ×{factor:.2f}" for number, factor in factors
            ) + ")"
        stats.morph = f"{area} · {round(amount * 100)}% · {summary}{area_note}"
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
# Video morph pipeline
# --------------------------------------------------------------------------- #


def _face_center_size(points: np.ndarray) -> tuple[float, float, float]:
    """Centre (x, y) and long side in pixels of a mesh's face oval."""
    oval = points[list(FACE_OVAL_INDICES)]
    x_min, y_min = oval.min(axis=0)
    x_max, y_max = oval.max(axis=0)
    return (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        max(x_max - x_min, y_max - y_min),
    )


@dataclass
class _FaceTrack:
    """Mutable tracker state for one face, shared by scan and processing.

    ``center_x``/``center_y``/``size`` are the *last known* position the
    matcher keeps updating; ``first_*`` freeze where the face was first
    seen, which the processing pass uses to start following the same person
    from the beginning of the video; ``best_*`` keep the clearest sighting
    for the UI thumbnail (scan only).
    """

    center_x: float
    center_y: float
    size: float
    first_center_x: float
    first_center_y: float
    first_size: float
    first_frame: int
    last_frame: int
    sightings: int = 0
    best_points: np.ndarray | None = None
    best_preview: np.ndarray | None = None
    best_area: float = 0.0
    gallery: list[np.ndarray] | None = None
    confirmed: bool = False


def _match_faces_to_tracks(
    tracks: list[_FaceTrack],
    faces: Sequence[np.ndarray],
    frame_index: int,
    grace: int,
    allow_relink: bool = False,
) -> list[tuple[int, int]]:
    """Match detected faces onto tracked faces by position and appearance.

    Each face is greedily matched to at most one track — and each track to
    at most one face — preferring the closest pair. Cost combines normalised
    centre distance with the *best* cosine distance to any exemplar in the
    track's appearance gallery. The gallery stores one descriptor per viewed
    pose (frontal, three-quarter, profile), so the same person turning from
    front to side is still matched to their own track instead of being treated
    as a new face; the position gate stops a newcomer who happens to stand
    where someone left from stealing their id.

    A track cannot be matched before it first appeared (minus ``grace``
    frames). Unless ``allow_relink`` is set, a track that has been missing for
    more than ``grace`` frames is considered gone, so one person replacing
    another in the same spot becomes two faces instead of a wrong morph. With
    ``allow_relink=True`` a long-lost track can be reclaimed by a later
    detection that matches its appearance and position.

    Matched tracks move to their face's position and their gallery is updated
    with the new view, so the next frame follows the face as it moves and as
    its pose changes. Returns ``(track_index, face_index)`` pairs sorted by
    track index.
    """
    scored: list[tuple[float, int, int]] = []
    face_descs = [_landmark_descriptor(points) for points in faces]
    for face_index, points in enumerate(faces):
        face_x, face_y, face_size = _face_center_size(points)
        for track_index, track in enumerate(tracks):
            if frame_index < track.first_frame - grace:
                continue
            if not allow_relink and frame_index - track.last_frame > grace:
                continue
            reach = max(face_size, track.size) * SLOT_MATCH_RATIO
            distance = math.hypot(face_x - track.center_x, face_y - track.center_y)
            if distance > reach:
                continue
            position_cost = distance / reach
            face_desc = face_descs[face_index]
            if face_desc is not None and track.gallery:
                sig_cost = float(
                    min(_cosine_distance(face_desc, g) for g in track.gallery)
                )
            else:
                sig_cost = 0.5
            combined = 0.6 * sig_cost + 0.4 * position_cost
            scored.append((combined, face_index, track_index))

    scored.sort()
    used_faces: set[int] = set()
    used_tracks: set[int] = set()
    matches: list[tuple[int, int]] = []
    for score, face_index, track_index in scored:
        if face_index in used_faces or track_index in used_tracks:
            continue
        if score > MATCH_THRESHOLD:
            continue
        used_faces.add(face_index)
        used_tracks.add(track_index)
        points = faces[face_index]
        face_x, face_y, face_size = _face_center_size(points)
        track = tracks[track_index]
        track.center_x = face_x
        track.center_y = face_y
        track.size = face_size
        track.last_frame = frame_index
        track.sightings += 1
        if track.sightings >= CONFIRM_HITS:
            track.confirmed = True
        # Add this view to the track's gallery (deduplicated against existing views).
        face_desc = face_descs[face_index]
        if face_desc is not None:
            if track.gallery is None:
                track.gallery = [face_desc]
            else:
                near_duplicate = any(
                    _cosine_distance(face_desc, g) < 0.15 for g in track.gallery
                )
                if not near_duplicate:
                    track.gallery.append(face_desc)
                    if len(track.gallery) > GALLERY_SIZE:
                        track.gallery.pop(0)
        matches.append((track_index, face_index))
    matches.sort()
    return matches


def _scan_video_faces(
    path: Path,
    *,
    finder: str,
    max_width: int,
    progress: ProgressCallback | None = None,
) -> VideoFaceScan:
    """Scan sampled frames of a video and collect its distinct faces.

    Up to ``VIDEO_SCAN_SAMPLES`` frames — evenly spread across the file when
    the container reports its length, otherwise every 15th frame — are
    landmarked with the same finder the morph will use. Faces are grouped
    into tracks by position and timing (see :func:`_match_faces_to_tracks`);
    each track becomes one numbered slot for the UI, keeping the clearest
    (largest) sighting's mesh and thumbnail plus where the face was first
    seen. The sampling step is recorded so the morph pass can allow the same
    slack when deciding when a face may appear.
    """
    capture = _open_capture(path)
    landmarker: FaceLandmarkerModel | None = None
    try:
        fps = _safe_fps(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ok, first = capture.read()
        if not ok or first is None:
            raise RuntimeError("No video frames could be decoded from this file.")
        out_width, out_height = _output_frame_size(first.shape[1], first.shape[0], max_width)

        if reported_frames > 0:
            count = min(VIDEO_SCAN_SAMPLES, reported_frames)
            span = max(1, count - 1)
            indices = sorted(
                {
                    min(reported_frames - 1, round(order * (reported_frames - 1) / span))
                    for order in range(count)
                }
            )
            step = max(1, (reported_frames - 1) // span)
        else:
            # Unknown length: walk forward, sampling every 15th frame.
            indices = list(range(0, VIDEO_SCAN_SAMPLES * 15, 15))
            step = 15

        if progress is not None:
            progress(0.2, "Loading the face landmarker…")
        landmarker = build_landmarker(progress, finder=finder)
        grace = max(2 * step, round(fps))

        tracks: list[_FaceTrack] = []
        samples = 0
        for order, index in enumerate(indices):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            samples += 1
            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(
                    frame, (out_width, out_height), interpolation=cv2.INTER_AREA
                )

            if progress is not None:
                progress(
                    0.3 + 0.65 * (order + 1) / len(indices),
                    f"Scanning for faces… frame {index + 1} ({samples} sampled)",
                )
            meshes = landmarker.find_faces(frame)
            matches = _match_faces_to_tracks(tracks, meshes, index, grace, allow_relink=True)
            for track_index, face_index in matches:
                track = tracks[track_index]
                points = meshes[face_index]
                area = FaceLandmarkerModel._mesh_area(points)
                if area > track.best_area:
                    track.best_area = area
                    track.best_points = points.copy()
                    track.best_preview = _face_crop_preview(frame, points)

            # Faces nobody claims become new tracks — genuinely new people
            # that could not be relinked to an existing track by appearance.
            matched_faces = {face_index for _, face_index in matches}
            for face_index, points in enumerate(meshes):
                if face_index in matched_faces or len(tracks) >= MAX_MORPH_FACES:
                    continue
                center_x, center_y, size = _face_center_size(points)
                tracks.append(
                    _FaceTrack(
                        center_x=center_x,
                        center_y=center_y,
                        size=size,
                        first_center_x=center_x,
                        first_center_y=center_y,
                        first_size=size,
                        first_frame=index,
                        last_frame=index,
                        sightings=1,
                        best_points=points.copy(),
                        best_preview=_face_crop_preview(frame, points),
                        best_area=FaceLandmarkerModel._mesh_area(points),
                        gallery=(
                            [_landmark_descriptor(points)]
                            if _landmark_descriptor(points) is not None
                            else []
                        ),
                    )
                )

        # Merge tracks that are the same person recorded twice (e.g. seen
        # first in profile/partial view, then front-on later). Tracks with
        # very similar landmark signatures and non-overlapping observation
        # windows are collapsed into one, keeping the clearest sighting.
        merged: list[_FaceTrack] = []
        for track in sorted(tracks, key=lambda item: item.best_area, reverse=True):
            target: _FaceTrack | None = None
            for kept in merged:
                overlap = not (
                    track.last_frame < kept.first_frame - grace
                    or kept.last_frame < track.first_frame - grace
                )
                if overlap:
                    continue
                if (
                    track.gallery
                    and kept.gallery
                    and min(
                        _cosine_distance(a, b)
                        for a in track.gallery
                        for b in kept.gallery
                    )
                    <= 0.30
                ):
                    target = kept
                    break
            if target is None:
                merged.append(track)
                continue
            if track.best_area > target.best_area:
                target.best_points = track.best_points
                target.best_preview = track.best_preview
                target.best_area = track.best_area
            if track.first_frame < target.first_frame:
                target.first_frame = track.first_frame
                target.first_center_x = track.first_center_x
                target.first_center_y = track.first_center_y
                target.first_size = track.first_size
            target.last_frame = max(target.last_frame, track.last_frame)
            target.sightings += track.sightings
            if track.gallery:
                if target.gallery is None:
                    target.gallery = []
                for desc in track.gallery:
                    if not any(
                        _cosine_distance(desc, g) < 0.15 for g in target.gallery
                    ):
                        target.gallery.append(desc)
                if len(target.gallery) > GALLERY_SIZE:
                    target.gallery = target.gallery[:GALLERY_SIZE]
        tracks = merged

        slots = [
            VideoFaceSlot(
                points=track.best_points,
                preview=track.best_preview,
                center_x=track.first_center_x,
                center_y=track.first_center_y,
                size=track.first_size,
                first_frame=track.first_frame,
                last_frame=track.last_frame,
                sightings=track.sightings,
            )
            for track in tracks
        ]
        slots.sort(key=lambda slot: FaceLandmarkerModel._mesh_area(slot.points), reverse=True)
        return VideoFaceScan(
            slots=slots,
            samples=samples,
            frames=reported_frames,
            step=step,
            width=out_width,
            height=out_height,
            fps=fps,
        )
    finally:
        capture.release()
        if landmarker is not None:
            landmarker.close()


def process_morph_video(
    source: Path,
    destination: Path,
    scan: VideoFaceScan,
    target_payloads: Sequence[bytes | None],
    *,
    amount: float,
    area: str,
    stride: int,
    max_width: int,
    crf: int,
    preset: str,
    keep_audio: bool,
    finder: str = MORPH_FINDER_MEDIAPIPE,
    progress: ProgressCallback | None = None,
) -> ProcessStats:
    """Morph every tracked face of ``source`` toward its target, per frame.

    ``scan`` comes from :func:`_scan_video_faces` run on the same file (the
    route caches it), so the tracked faces — and the numbering the user saw
    when picking target pictures — are exactly the ones processed here.
    ``target_payloads`` holds one entry per tracked face, largest first; a
    ``None`` entry leaves that face untouched throughout. Each target image
    is decoded and its own face landmarked once, then every frame re-runs
    the image-morph pipeline (:func:`_morph_face_pair`) on the face meshes
    found in that frame: the mixed face replaces the source face inside its
    feathered oval and everything else in the frame — background, framing,
    unpaired faces — is preserved. Faces are matched to the tracked slots by
    position and timing (see :func:`_match_faces_to_tracks`); a face the
    matcher cannot follow is simply left alone.

    ``stride`` re-uses the last frame's meshes between detections exactly
    like the blur pipeline, so 1 keeps morphs glued to moving faces. The
    result is an H.264/yuv420p MP4 with the original audio copied back in
    when ``keep_audio`` is set.
    """
    if finder not in (MORPH_FINDER_MEDIAPIPE, MORPH_FINDER_YUNET):
        finder = MORPH_FINDER_MEDIAPIPE
    stats = ProcessStats(
        detector=(
            f"{DETECTOR_YUNET} + {DETECTOR_LANDMARKER}"
            if finder == MORPH_FINDER_YUNET
            else DETECTOR_LANDMARKER
        ),
    )
    started = time.perf_counter()
    capture = _open_capture(source)
    sink: VideoSink | None = None
    landmarker: FaceLandmarkerModel | None = None

    try:
        payload_list = list(target_payloads)[: len(scan.slots)]
        if len(payload_list) < len(scan.slots):
            payload_list += [None] * (len(scan.slots) - len(payload_list))
        if all(payload is None for payload in payload_list):
            raise RuntimeError("Provide a target image for at least one tracked face.")

        fps = _safe_fps(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("No video frames could be decoded from this file.")
        out_width, out_height = _output_frame_size(frame.shape[1], frame.shape[0], max_width)
        if (out_width, out_height) != (scan.width, scan.height):
            raise RuntimeError(
                "The video no longer matches its scanned faces — upload it again."
            )

        if progress is not None:
            progress(0.08, "Loading the face landmarker…")
        landmarker = build_landmarker(progress, finder=finder)

        # Decode each target once: re-running this per frame would dominate
        # the run time on longer videos.
        targets: list[tuple[np.ndarray, np.ndarray] | None] = []
        for slot_index, payload in enumerate(payload_list):
            if payload is None:
                targets.append(None)
                continue
            label = f"target for face {slot_index + 1}"
            try:
                target = _decode_image_bgr(payload, label)
                if max_width:
                    target = _limit_width(target, max_width)
                target_points = landmarker.largest_face(target, label)
            except RuntimeError as exc:
                raise RuntimeError(f"Face {slot_index + 1}: {exc}") from exc
            targets.append((target, target_points))

        paired = [index for index, target in enumerate(targets) if target is not None]

        # Whole-image blending replaces the entire frame with the aligned
        # target, which only makes sense for a single morphed face; with
        # several faces every morph stays inside its own oval so the faces
        # do not overwrite each other.
        whole_image = area == MORPH_WHOLE_IMAGE and len(paired) == 1
        area_note = (
            " · whole image applies to a single face; blended per face oval"
            if area == MORPH_WHOLE_IMAGE and len(paired) > 1
            else ""
        )

        sink = make_sink(
            destination,
            out_width,
            out_height,
            fps,
            crf=crf,
            preset=preset,
            audio_source=source if keep_audio else None,
        )

        grace = max(2 * scan.step, round(scan.fps))
        tracks = [
            _FaceTrack(
                center_x=slot.center_x,
                center_y=slot.center_y,
                size=slot.size,
                first_center_x=slot.center_x,
                first_center_y=slot.center_y,
                first_size=slot.size,
                first_frame=slot.first_frame,
                last_frame=slot.first_frame,
                gallery=(
                    [_landmark_descriptor(slot.points)]
                    if _landmark_descriptor(slot.points) is not None
                    else []
                ),
            )
            for slot in scan.slots
        ]

        amount = float(np.clip(amount, 0.0, 1.0))
        factor_sums = [0.0] * len(scan.slots)
        factor_counts = [0] * len(scan.slots)
        assignments: list[tuple[int, np.ndarray]] = []
        frame_index = 0
        stride = max(1, int(stride))

        if progress is not None:
            progress(0.15, "Morphing faces…")

        while True:
            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(
                    frame, (out_width, out_height), interpolation=cv2.INTER_AREA
                )

            if frame_index % stride == 0:
                faces = landmarker.find_faces(frame)
                stats.detection_frames += 1
                assignments = [
                    (track_index, faces[face_index])
                    for track_index, face_index in _match_faces_to_tracks(
                        tracks, faces, frame_index, grace
                    )
                    if targets[track_index] is not None
                ]
                if assignments:
                    stats.faces += len(assignments)
                    stats.frames_with_faces += 1

            for track_index, points in assignments:
                target_image, target_points = targets[track_index]
                frame, factor = _morph_face_pair(
                    frame, points, target_image, target_points, amount, whole_image
                )
                factor_sums[track_index] += factor
                factor_counts[track_index] += 1

            sink.write(frame)
            frame_index += 1

            if progress is not None and (
                frame_index % 5 == 0 or frame_index == reported_frames
            ):
                if reported_frames > 0:
                    fraction = min(0.97, 0.15 + 0.82 * frame_index / reported_frames)
                    progress(
                        fraction, f"Morphing faces… {frame_index}/{reported_frames} frames"
                    )
                else:
                    progress(0.5, f"Morphing faces… {frame_index} frames")

            ok, frame = capture.read()
            if not ok:
                break

        if progress is not None:
            progress(0.97, f"Finalising… {frame_index} frames encoded")

        morphed = [index for index in paired if factor_counts[index] > 0]
        summary = (
            f"{len(morphed)} of {len(scan.slots)} tracked "
            f"face{'s' if len(scan.slots) != 1 else ''} morphed"
        )
        if morphed:
            summary += " (" + ", ".join(
                f"face {index + 1}: ×{factor_sums[index] / factor_counts[index]:.2f}"
                for index in morphed
            ) + ")"
        stats.frames = frame_index
        stats.width = out_width
        stats.height = out_height
        stats.fps = fps
        stats.elapsed = time.perf_counter() - started
        stats.encoder = sink.encoder_name
        stats.audio = sink.audio_status
        stats.morph = f"{area} · {round(amount * 100)}% · {summary}{area_note}"
        return stats
    finally:
        capture.release()
        try:
            if sink is not None:
                sink.close()
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
            "```\n./.venv/bin/streamlit run v10.py\n```"
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
    is_video = kind in (KIND_VIDEO, KIND_VIDEO_MORPH)
    is_morph = kind in (KIND_MORPH, KIND_VIDEO_MORPH)
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
                    "similar composition. With more than one morphed face, each "
                    "blend stays inside its own face oval so the faces do not "
                    "overwrite each other."
                ),
            )
            finder = st.radio(
                "Face finder",
                (MORPH_FINDER_MEDIAPIPE, MORPH_FINDER_YUNET),
                help=(
                    "How the face is located before morphing. MediaPipe "
                    "FaceLandmarker finds the faces and their 478-point meshes "
                    "in one pass — most accurate when each face fills a fair "
                    "share of the frame — with a YuNet sweep adding any faces "
                    "its own detector missed. YuNet instead locates the face "
                    "boxes first and each mesh is fitted on an optimally sized "
                    "crop, which is more reliable when the faces are small in "
                    "the photo. Either way the mesh always comes from "
                    "MediaPipe, and the other finder is tried as a fallback."
                ),
            )
            if is_video:
                st.divider()
                st.header("Detection")
                stride = st.select_slider(
                    "Detect every N frames",
                    options=(1, 2, 3, 4),
                    value=1,
                    help=(
                        "Faces are re-detected every N frames and each morph "
                        "keeps its last position in between, so 2–3 is much "
                        "faster with a little lag on fast movement."
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
                f"Faces are found with {finder}."
            )
            settings: dict[str, object] = {
                "max_width": 0 if width_choice == "Original" else int(width_choice.split()[0]),
                "morph_amount": amount / 100.0,
                "morph_area": area,
                "morph_finder": finder,
            }
            if is_video:
                settings.update(
                    stride=stride,
                    quality=quality,
                    keep_audio=keep_audio,
                )
            return settings

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
        if stats.faces == 0:
            if is_morph:
                st.warning(
                    "No faces were morphed. Provide a target image for at least "
                    "one detected face, or raise the morph amount above 0%."
                )
            else:
                st.warning(
                    "No faces were detected, so the image was only re-encoded. "
                    "Try lowering the detection confidence or switching detectors."
                )
        return

    is_video_morph = result.kind == KIND_VIDEO_MORPH
    face_ratio = (stats.frames_with_faces / stats.frames * 100) if stats.frames else 0.0
    columns = st.columns(4)
    columns[0].metric("Frames", f"{stats.frames:,}")
    columns[1].metric(
        "Faces morphed" if is_video_morph else "Faces blurred", f"{stats.faces:,}"
    )
    columns[2].metric(
        "Frames with a morph" if is_video_morph else "Frames with a face",
        f"{face_ratio:.0f}%",
    )
    columns[3].metric("Processing time", f"{stats.elapsed:.1f}s")

    caption = (
        f"{stats.width}×{stats.height} · {stats.fps:.2f} fps · "
        f"{stats.detector} · {stats.encoder} · audio: {stats.audio}"
    )
    if is_video_morph and stats.morph != "none":
        caption += f" · morph: {stats.morph}"
    st.caption(caption)

    if stats.faces == 0:
        st.warning(
            "No faces were morphed. Provide a target image for at least one "
            "tracked face, or check that the video shows clear, reasonably "
            "large faces."
            if is_video_morph
            else "No faces were detected, so the video was only re-encoded. "
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
            "Download morphed video"
            if result.kind == KIND_VIDEO_MORPH
            else "Download blurred video",
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


# Uploaded files larger than ~200 MB would exceed Streamlit's default
# ``server.maxUploadSize`` and ``uploaded.getvalue()`` would load the whole
# file into RAM. Stream the upload to disk in 1 MiB chunks instead so
# multi-hundred-MB videos can be processed without holding them in memory.
_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB


def _stream_upload_to_file(
    uploaded: st.runtime.uploaded_file_manager.UploadedFile,
    destination: Path,
) -> int:
    """Stream an UploadedFile to ``destination`` in chunks, returning total bytes."""
    uploaded.seek(0)
    total = 0
    with destination.open("wb") as out:
        while True:
            chunk = uploaded.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    uploaded.seek(0)
    return total


def _validate_image_size(
    uploaded: st.runtime.uploaded_file_manager.UploadedFile | None,
    label: str = "image",
) -> bool:
    """Reject image uploads larger than IMAGE_MAX_SIZE.

    Returns True when the file is acceptable (or None), False when it is
    too large and an error has already been shown to the user.
    """
    if uploaded is None:
        return True
    size = getattr(uploaded, "size", None)
    if size is None:
        return True
    if size > IMAGE_MAX_SIZE:
        st.error(
            f"{label} '{uploaded.name}' is {size / 1e6:.1f} MB, which exceeds "
            f"the {IMAGE_MAX_SIZE / 1e6:.0f} MB image limit. "
            "Videos may be up to 2 GB, but images are capped at 20 MB."
        )
        return False
    return True


def _run_video(uploaded: st.runtime.uploaded_file_manager.UploadedFile, settings: dict[str, object]) -> None:
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
            _stream_upload_to_file(uploaded, source)
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
            source=(uploaded.name, int(getattr(uploaded, "size", source.stat().st_size))),
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


def _morph_source_faces(
    source: st.runtime.uploaded_file_manager.UploadedFile,
    settings: dict[str, object],
) -> list[np.ndarray]:
    """Detect every face in the morph source image, cached across reruns.

    Streamlit re-runs the script on every widget interaction, so the meshes
    are cached in session state keyed by the upload and the finder settings;
    the landmarker only runs when the source image (or the settings) actually
    change.
    """
    key = (
        source.name,
        source.size,
        str(settings["morph_finder"]),
        int(settings["max_width"]),
    )
    cache: dict[str, object] | None = st.session_state.get("morph_detection")
    if cache is not None and cache["key"] == key:
        return cache["faces"]  # type: ignore[no-any-return]

    with st.spinner("Detecting faces in the source image…"):
        image = _decode_image_bgr(source.getvalue(), "source")
        max_width = int(settings["max_width"])
        if max_width:
            image = _limit_width(image, max_width)
        landmarker = build_landmarker(None, finder=str(settings["morph_finder"]))
        try:
            faces = landmarker.all_faces(image, "source")
        finally:
            landmarker.close()

    st.session_state["morph_detection"] = {"key": key, "faces": faces}
    return faces


def _face_crop_preview(
    image: np.ndarray, points: np.ndarray, long_side: int = 260
) -> np.ndarray:
    """RGB thumbnail of one detected face, used to label its target uploader.

    Takes the internal BGR image that processing works on and returns an
    RGB thumbnail ready for ``st.image``: the screen always shows the photo
    with its true colours, while everything under the hood stays in OpenCV's
    BGR order.
    """
    x0, y0, x1, y1 = _face_roi(points, image.shape, margin=0.35)
    crop = image[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    if max(height, width) >= 2:
        scale = long_side / max(height, width)
        if abs(scale - 1.0) >= 1e-3:
            crop = cv2.resize(
                crop,
                (max(2, round(width * scale)), max(2, round(height * scale))),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _run_morph(
    source: st.runtime.uploaded_file_manager.UploadedFile,
    targets: list[st.runtime.uploaded_file_manager.UploadedFile | None],
    settings: dict[str, object],
) -> None:
    if all(target is None for target in targets):
        st.warning("Provide a target image for at least one detected face.")
        return

    source_payload = source.getvalue()
    target_payloads: list[bytes | None] = [
        target.getvalue() if target is not None else None for target in targets
    ]
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
            target_payloads,
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
                tuple(
                    (target.name, target.size) if target is not None else None
                    for target in targets
                ),
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


def _video_morph_face_slots(
    source: st.runtime.uploaded_file_manager.UploadedFile,
    settings: dict[str, object],
) -> VideoFaceScan:
    """Scan the morph source video for distinct faces, cached across reruns.

    Streamlit re-runs the script on every widget interaction, so the scan —
    which decodes sampled frames and runs the landmarker — is cached in
    session state keyed by the upload and the finder settings; it only
    repeats when the video (or the settings) actually change.
    """
    key = (
        source.name,
        source.size,
        str(settings["morph_finder"]),
        int(settings["max_width"]),
    )
    cache: dict[str, object] | None = st.session_state.get("video_morph_scan")
    if cache is not None and cache["key"] == key:
        return cache["scan"]  # type: ignore[no-any-return]

    with st.spinner("Scanning the video for faces…"):
        with tempfile.TemporaryDirectory(prefix="faceblur_") as workspace:
            video_path = Path(workspace) / "input_video"
            _stream_upload_to_file(source, video_path)
            scan = _scan_video_faces(
                video_path,
                finder=str(settings["morph_finder"]),
                max_width=int(settings["max_width"]),
            )

    st.session_state["video_morph_scan"] = {"key": key, "scan": scan}
    return scan


def _run_video_morph(
    source: st.runtime.uploaded_file_manager.UploadedFile,
    targets: list[st.runtime.uploaded_file_manager.UploadedFile | None],
    scan: VideoFaceScan,
    settings: dict[str, object],
) -> None:
    if all(target is None for target in targets):
        st.warning("Provide a target image for at least one tracked face.")
        return

    target_payloads: list[bytes | None] = [
        target.getvalue() if target is not None else None for target in targets
    ]
    stem = Path(source.name).stem or "video"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_ ").strip() or "video"
    output_name = f"{safe_stem}_morph.mp4"

    progress_bar = st.progress(0.0, text="Starting…")
    status = st.status("Morphing faces in the video…", expanded=True)

    def report(fraction: float, message: str) -> None:
        progress_bar.progress(max(0.0, min(1.0, fraction)), text=message)
        status.update(label=message)

    try:
        with tempfile.TemporaryDirectory(prefix="faceblur_") as workspace:
            source_path = Path(workspace) / "input_video"
            _stream_upload_to_file(source, source_path)
            destination = Path(workspace) / output_name

            stats = process_morph_video(
                source_path,
                destination,
                scan,
                target_payloads,
                amount=float(settings["morph_amount"]),
                area=str(settings["morph_area"]),
                stride=int(settings["stride"]),
                max_width=int(settings["max_width"]),
                crf=QUALITY_CRF[str(settings["quality"])],
                preset=QUALITY_PRESET[str(settings["quality"])],
                keep_audio=bool(settings["keep_audio"]),
                finder=str(settings["morph_finder"]),
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
            source=(
                source.name,
                source.size,
                tuple(
                    (target.name, target.size) if target is not None else None
                    for target in targets
                ),
            ),
            kind=KIND_VIDEO_MORPH,
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
    action = "Morph faces" if kind in (KIND_MORPH, KIND_VIDEO_MORPH) else "Blur faces"
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
    if not _validate_image_size(uploaded, "Source image"):
        return
    if not _validate_image_size(background, "Background image"):
        return
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
    source = st.file_uploader(
        "Source image (the base)",
        type=list(IMAGE_EXTENSIONS),
        key="morph_source",
        help=(
            "Every face found in this image can receive its own target picture. "
            "Faces without a target stay untouched. The result keeps this "
            "image's size, framing and background."
        ),
    )

    if source is None:
        _forget_result(KIND_MORPH)
        st.info(
            "Upload a source image to get started; every face in it can be "
            "morphed with its own target picture. Nothing leaves your computer."
        )
        with st.expander("How face morphing works"):
            st.markdown(
                """
- Upload a source image first. **All faces** in it are detected — each
  shown as a numbered thumbnail beside its own target uploader — using the
  chosen **finder**: MediaPipe FaceLandmarker in one pass (with a YuNet
  sweep for faces its own detector misses), or YuNet first when faces are
  small in the photo. Models are downloaded once and cached locally.
- For each numbered face you can upload a **separate target picture**; it
  is previewed the moment it is uploaded, and the largest face in it is
  blended into that source face. **Faces without a target image are left
  completely untouched.**
- For every paired face: the **ROI** is determined in both photos in a
  roll-normalised frame, so a tilted head reports the same ROI as an upright
  one. The whole target image is resized with a single uniform,
  proportionate scale so its face ROI matches the source face's in pixel
  size.
- The resized target face is placed **exactly** on the source face's
  footprint — same size, position and tilt, anchored on the eye midpoint —
  derived directly from the two ROIs rather than fitted to landmarks, so the
  morph stays proportionate at every amount.
- Both faces are warped to an in-between shape with a Delaunay-triangulated
  piecewise affine warp, then cross-dissolved into one **mixed face**. The
  target's skin tone is matched to the source so it does not look pasted.
- Each mixed face is placed back on the **source image** inside a softly
  feathered face oval of its own — the background, framing and all unpaired
  faces stay untouched.
- **Morph amount** blends from the source (0%) to the target (100%) and is
  shared by every morphed face.
- **Face only** keeps the source background. **Whole image** also blends the
  aligned target background — it applies when a single face is morphed; with
  several faces each blend stays inside its own oval so the faces do not
  overwrite each other.
- Use clear, front-facing photos where the faces are reasonably large.
                """
            )
        return

    if not _validate_image_size(source, "Source image"):
        return

    try:
        faces = _morph_source_faces(source, settings)
        image = _decode_image_bgr(source.getvalue(), "source")
        max_width = int(settings["max_width"])
        if max_width:
            image = _limit_width(image, max_width)
    except Exception as exc:  # noqa: BLE001 - show why detection failed
        _forget_result(KIND_MORPH)
        st.error(f"{exc}")
        return

    count = len(faces)
    st.subheader(f"{count} face{'s' if count != 1 else ''} detected in the source image")
    # The upload is shown exactly as uploaded; detection and processing work
    # on the internal BGR copy, which is never shown to the user.
    st.image(source.getvalue(), caption=source.name)
    st.caption(f"{source.size / 1e6:.2f} MB · {source.name}")

    st.subheader("Target image per face")
    st.caption(
        "The largest face in each target picture is blended into the source "
        "face shown beside its uploader. Leave an uploader empty to keep "
        "that face unchanged."
    )
    targets: list[st.runtime.uploaded_file_manager.UploadedFile | None] = []
    for index, points in enumerate(faces):
        crop = _face_crop_preview(image, points)
        columns = st.columns([1, 3])
        with columns[0]:
            st.image(crop, caption=f"Face {index + 1}")
        with columns[1]:
            target = st.file_uploader(
                f"Target image for face {index + 1}",
                type=list(IMAGE_EXTENSIONS),
                key=f"morph_target_{index}",
                help=(
                    "The largest face in this picture is blended into the "
                    "numbered source face. Leave it empty to keep this face "
                    "unchanged."
                ),
            )
            if target is not None:
                if not _validate_image_size(target, f"Target image for face {index + 1}"):
                    targets.append(None)
                    continue
                # Previewed the moment it is uploaded, so the user can
                # confirm the right picture went to the right face.
                st.image(
                    target.getvalue(),
                    caption=f"{target.name} — blended into face {index + 1}",
                    width=340,
                )
                st.caption(f"{target.size / 1e6:.2f} MB · {target.name}")
        targets.append(target)

    provided = sum(1 for target in targets if target is not None)

    options_left, options_right = st.columns([3, 1])
    with options_left:
        st.write(
            f"**Morph amount:** {round(float(settings['morph_amount']) * 100)}% · "
            f"**Blend area:** {settings['morph_area']} · "
            f"**Finder:** {settings['morph_finder']} · "
            f"**Targets provided:** {provided} of {count}"
        )
    with options_right:
        start = st.button(
            "Morph faces", type="primary", key="morph_faces", disabled=provided == 0
        )
    if provided == 0:
        st.info("Provide a target image for at least one face to enable morphing.")

    source_key = (
        source.name,
        source.size,
        tuple(
            (target.name, target.size) if target is not None else None
            for target in targets
        ),
    )

    if start:
        # Rendered outside the column so progress and errors span the page.
        _run_morph(source, targets, settings)

    _show_result(source_key, KIND_MORPH, settings)


def _video_morph_route(settings: dict[str, object]) -> None:
    source = st.file_uploader(
        "Source video (the base)",
        type=list(VIDEO_EXTENSIONS),
        key="morph_video_source",
        help=(
            "Every face tracked in this video can receive its own target "
            "picture. Faces without a target stay untouched. The result keeps "
            "this video's framing, timing and audio."
        ),
    )

    if source is None:
        _forget_result(KIND_VIDEO_MORPH)
        st.info(
            "Upload a source video to get started; every face tracked in it "
            "can be morphed with its own target picture. Nothing leaves your "
            "computer."
        )
        with st.expander("How video face morphing works"):
            st.markdown(
                """
- Upload a source video first. Evenly spread frames are scanned to learn
  **which distinct faces** appear in it — each face is followed by position
  and timing as it moves, appears and leaves — and every face gets a
  numbered thumbnail beside its own target uploader.
- For each numbered face you can upload a **separate target picture**; it
  is previewed the moment it is uploaded, and the largest face in it is
  blended into that tracked face in every frame where it is found.
  **Faces without a target image are left completely untouched.**
- Each frame then runs the same pipeline as image morphing: the face ROI
  is matched in a roll-normalised frame, the whole target is scaled so
  its ROI matches, both faces are warped to an in-between shape with a
  Delaunay-triangulated piecewise warp and cross-dissolved — with the
  target's skin tone matched to the source — and the mixed face is placed
  back inside a feathered oval. Everything else in the frame is preserved.
- Faces are matched to their targets **by position**: the morph follows
  each face as it moves. If two people cross paths the targets can follow
  the position rather than the person, and a face that appears only
  briefly between scanned frames is picked up once it comes near a
  tracked position.
- **Detect every N frames** (sidebar) re-detects the faces and reuses the
  last position in between; 1 keeps the morph glued to moving faces.
- The result is an H.264 MP4 with the original audio copied back in.
                """
            )
        return

    try:
        scan = _video_morph_face_slots(source, settings)
    except Exception as exc:  # noqa: BLE001 - show why the scan failed
        _forget_result(KIND_VIDEO_MORPH)
        st.error(f"{exc}")
        return

    count = len(scan.slots)
    st.subheader(
        f"{count} face{'s' if count != 1 else ''} tracked in the source video"
    )
    # The upload is shown exactly as uploaded; scanning and processing work
    # on the internal BGR frames, which are never shown to the user.
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        if source.name.lower().endswith((".mp4", ".mov", ".m4v", ".webm")):
            source.seek(0)
            st.video(source)
        else:
            st.info(
                f"{source.name} cannot be previewed in the browser, "
                "but it can be processed."
            )
        st.caption(f"{source.size / 1e6:.1f} MB · {source.name}")
    with right:
        st.subheader("Options")
        st.write(
            f"**Morph amount:** {round(float(settings['morph_amount']) * 100)}% · "
            f"**Blend area:** {settings['morph_area']} · "
            f"**Finder:** {settings['morph_finder']} · "
            f"**Detect every {settings['stride']} frame(s)**"
        )

    if count == 0:
        _forget_result(KIND_VIDEO_MORPH)
        st.warning(
            "No faces were found in the scanned frames of this video. Try "
            "the other face finder in the sidebar, or check that the faces "
            "are clear and reasonably large."
        )
        return

    st.subheader("Target image per face")
    st.caption(
        "The largest face in each target picture is blended into the tracked "
        "face shown beside its uploader, in every frame where it is found. "
        "Leave an uploader empty to keep that face unchanged."
    )
    targets: list[st.runtime.uploaded_file_manager.UploadedFile | None] = []
    for index, slot in enumerate(scan.slots):
        columns = st.columns([1, 3])
        with columns[0]:
            st.image(slot.preview, caption=f"Face {index + 1}")
            st.caption(f"seen in {slot.sightings} of {scan.samples} scanned frames")
        with columns[1]:
            target = st.file_uploader(
                f"Target image for face {index + 1}",
                type=list(IMAGE_EXTENSIONS),
                key=f"video_morph_target_{index}",
                help=(
                    "The largest face in this picture is blended into the "
                    "numbered tracked face. Leave it empty to keep this face "
                    "unchanged."
                ),
            )
            if target is not None:
                if not _validate_image_size(target, f"Target image for face {index + 1}"):
                    targets.append(None)
                    continue
                # Previewed the moment it is uploaded, so the user can
                # confirm the right picture went to the right face.
                st.image(
                    target.getvalue(),
                    caption=f"{target.name} — blended into face {index + 1}",
                    width=340,
                )
                st.caption(f"{target.size / 1e6:.2f} MB · {target.name}")
        targets.append(target)

    provided = sum(1 for target in targets if target is not None)

    options_left, options_right = st.columns([3, 1])
    with options_left:
        st.write(f"**Targets provided:** {provided} of {count}")
    with options_right:
        start = st.button(
            "Morph faces",
            type="primary",
            key="morph_video_faces",
            disabled=provided == 0,
        )
    if provided == 0:
        st.info("Provide a target image for at least one tracked face to enable morphing.")

    source_key = (
        source.name,
        source.size,
        tuple(
            (target.name, target.size) if target is not None else None
            for target in targets
        ),
    )

    if start:
        # Rendered outside the column so progress and errors span the page.
        _run_video_morph(source, targets, scan, settings)

    _show_result(source_key, KIND_VIDEO_MORPH, settings)


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
        "Images can also have their background removed or replaced, and faces — in "
        "photos and in videos — can be morphed toward a target picture. Built with "
        "OpenCV 5, MediaPipe Tasks and Streamlit."
    )

    kind = st.radio(
        "What would you like to do?",
        (KIND_IMAGE, KIND_VIDEO, KIND_MORPH, KIND_VIDEO_MORPH),
        index=None,
        horizontal=True,
        help="Pick a type first; the uploader and options below adapt to it.",
    )
    if kind is None:
        st.info(
            "Choose **Image**, **Video**, **Face morph** or **Video morph** above "
            "to get started. Nothing leaves your computer."
        )
        st.caption(
            "Images are blurred and re-encoded in memory, with optional background "
            "removal or replacement. Videos are re-encoded to H.264 with the original "
            "audio copied back in. Face morph blends a source and a target photo; "
            "video morph does the same for every tracked face in a video."
        )
        return

    st.caption(
        "Image mode: blur, then optionally remove the background or replace it with "
        "your own image."
        if kind == KIND_IMAGE
        else (
            "Video mode: the result is H.264 with the original audio copied back in."
            if kind == KIND_VIDEO
            else (
                "Face morph mode: upload a source and a target image, then blend "
                "the largest face from each."
                if kind == KIND_MORPH
                else "Video morph mode: upload a video and one target image per "
                "tracked face, then blend them frame by frame."
            )
        )
    )

    settings = _sidebar(str(kind))

    if kind == KIND_IMAGE:
        _image_route(settings)
    elif kind == KIND_VIDEO:
        _video_route(settings)
    elif kind == KIND_MORPH:
        _morph_route(settings)
    else:
        _video_morph_route(settings)


if __name__ == "__main__":
    main()
