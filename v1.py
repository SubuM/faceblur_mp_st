"""FaceBlur Studio — a modern rewrite of ``streamlit_app.py``.

The original app relied on APIs that no longer exist in the current, free
package releases:

* ``mediapipe.solutions.face_detection`` was removed in MediaPipe 0.10.31 and is
  absent from MediaPipe 1.x. Only the Tasks API (``mediapipe.tasks.python``)
  remains, and models are no longer bundled with the wheel.
* ``st.set_page_config(page_icon="timer_clock")`` is not a valid icon value.
* OpenCV 5 moved ``CascadeClassifier`` out of the core ``objdetect`` module.
* ``mp4v`` video files are not playable in browsers, so the download was fine
  but the Streamlit preview never worked.

This file keeps the same idea (upload a video, blur every face, download the
result) but uses only currently maintained, completely free components:

* Streamlit >= 1.50 for the UI (``st.status``, ``st.progress``, session state,
  ``download_button(on_click="ignore")`` so a download no longer re-runs the
  whole pipeline).
* OpenCV 5 + ``cv2.FaceDetectorYN`` (YuNet, BSD/MIT, ONNX) as the default
  detector — no ``opencv-contrib`` specific model is required and it is more
  accurate than BlazeFace for arbitrary videos.
* MediaPipe Tasks ``FaceDetector`` with ``blaze_face_short_range.tflite``
  (Apache-2.0) as an alternative backend. The model is downloaded once and
  cached, because MediaPipe stopped shipping models in the wheel.
* PyAV (LGPL FFmpeg bindings) to encode H.264/yuv420p — the only MP4 flavour
  that actually plays in browsers — and to copy the original audio track back
  in. ``imageio-ffmpeg`` is a drop-in alternative, but PyAV avoids spawning a
  subprocess per job.
* No paid APIs, no cloud calls, no API keys. Models come from Google Cloud
  Storage and the OpenCV Zoo and run locally on CPU.

Run with::

    pip install -r requirements-v1.txt
    streamlit run v1.py
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
VIDEO_EXTENSIONS = ("mp4", "mov", "m4v", "avi", "mkv", "webm", "mpeg", "mpg")

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

DETECTOR_YUNET = "YuNet (OpenCV DNN)"
DETECTOR_MEDIAPIPE = "MediaPipe BlazeFace"

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


@dataclass
class ProcessingResult:
    data: bytes
    filename: str
    stats: ProcessStats
    settings: dict[str, object] = field(default_factory=dict)
    source: tuple[str, int] = ("", 0)


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
            "```\n./.venv/bin/streamlit run v1.py\n```"
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
                "YuNet (OpenCV Zoo) and MediaPipe BlazeFace models are free to use."
            )
        },
    )


def _sidebar() -> dict[str, object]:
    with st.sidebar:
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
        smooth = st.checkbox(
            "Temporal smoothing",
            value=True,
            help="Reduces the frame-to-frame jitter of the blur rectangle.",
        )

        st.divider()
        st.header("Output")
        quality = st.selectbox("Quality", tuple(QUALITY_CRF), index=1)
        width_choice = st.selectbox(
            "Maximum width",
            ("Original", "1920 px", "1280 px", "854 px"),
            help="Downscaling makes processing much faster for 4K footage.",
        )
        keep_audio = st.checkbox("Keep original audio", value=True)

        st.divider()
        st.caption(
            "100% local processing. No uploads to third-party services, no API keys. "
            f"PyAV encoder: {'available' if av is not None else 'missing (install av)'}."
        )

    return {
        "detector": detector,
        "confidence": confidence,
        "stride": stride,
        "blur_mode": blur_mode,
        "strength": strength,
        "margin": margin / 100.0,
        "feather": feather,
        "smooth": smooth,
        "quality": quality,
        "max_width": 0 if width_choice == "Original" else int(width_choice.split()[0]),
        "keep_audio": keep_audio,
    }


def _render_stats(result: ProcessingResult) -> None:
    stats = result.stats
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


def _run(uploaded: st.runtime.uploaded_file_manager.UploadedFile, settings: dict[str, object]) -> None:
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


def main() -> None:
    if not _check_runtime():
        st.stop()

    _page_config()

    st.title("🎭 FaceBlur Studio")
    st.caption(
        "Blur or pixelate every face in a video, entirely on your machine. "
        "Built with OpenCV 5, MediaPipe Tasks and Streamlit."
    )

    settings = _sidebar()

    uploaded = st.file_uploader(
        "Upload a video",
        type=list(VIDEO_EXTENSIONS),
        help="MP4/MOV work best. Other containers are converted to MP4.",
    )

    if uploaded is None:
        st.session_state.pop("result", None)
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
        start = st.button("Blur faces", type="primary")

    if start:
        # Rendered outside the column so the progress bar and errors span the page.
        _run(uploaded, settings)

    result: ProcessingResult | None = st.session_state.get("result")
    if result is not None and result.source == source_key:
        if result.settings != settings:
            st.info("Settings changed — click **Blur faces** to re-process with the new options.")
        _render_result(result)


if __name__ == "__main__":
    main()
