from __future__ import annotations
import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, NamedTuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Event, Lock, Manager, Process
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorRange(NamedTuple):
    """Immutable color range in HSV space"""
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class Rectangle:
    """Immutable rectangle with cached properties"""
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self):
        if any(v < 0 for v in (self.x, self.y, self.width, self.height)):
            raise ValueError("Rectangle dimensions must be non-negative")

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def overlaps(self, other: Rectangle) -> bool:
        return (self.x < other.x + other.width and
                self.x + self.width > other.x and
                self.y < other.y + other.height and
                self.y + self.height > other.y)


class CameraProcessor:
    """Optimized camera capture and image processing for robot arm control"""

    DEFAULT_FRAME_WIDTH = 640
    DEFAULT_FRAME_HEIGHT = 480
    DEFAULT_CNN_SIZE = (128, 128)
    COLOR_RANGES: Dict[str, ColorRange] = {
        'pink': ColorRange(lower=np.array([130, 34, 175], dtype=np.uint8),
                           upper=np.array([180, 255, 255], dtype=np.uint8)),
        'blue': ColorRange(lower=np.array([90, 130, 128], dtype=np.uint8),
                           upper=np.array([180, 255, 255], dtype=np.uint8))
    }

    def __init__(self, zoom_factor: float = 2.2, x_offset: int = 430, y_offset: int = 140, camera_id: int = 0, buffer_size: int = 3):
        self.zoom_factor = max(1.0, zoom_factor)
        self.x_offset = max(0, x_offset)
        self.y_offset = max(0, y_offset)
        self.buffer_size = buffer_size
        self.manager = Manager()
        self.frame_buffer = self.manager.list()
        self._running = Event()
        self._lock = Lock()
        self._running.set()
        self.executor = ProcessPoolExecutor(max_workers=4)
        self._init_camera(camera_id)
        self._start_capture_process()

    def _init_camera(self, camera_id: int) -> None:
        for attempt in range(3):
            try:
                self.cap = cv2.VideoCapture(camera_id)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open camera {camera_id}")

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                             self.DEFAULT_FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                             self.DEFAULT_FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                break
            except Exception as e:
                logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(1.0)

    def _start_capture_process(self) -> None:
        self.capture_process = Process(
            target=self._capture_loop, name="CaptureProcess", daemon=True)
        self.capture_process.start()

    @contextmanager
    def frame_lock(self):
        with self._lock:
            yield

    def _capture_loop(self) -> None:
        frames_processed = 0
        last_fps_time = time.time()

        while self._running.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.01)
                    continue

                frame = np.asarray(frame, dtype=np.uint8)
                with self.frame_lock():
                    zoomed_frame = self._zoom_camera(frame)
                    self.frame_buffer.append(zoomed_frame)
                    if len(self.frame_buffer) > self.buffer_size:
                        self.frame_buffer.pop(0)

                frames_processed += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frames_processed / (current_time - last_fps_time)
                    logger.debug(f"Camera FPS: {fps:.2f}")
                    frames_processed = 0
                    last_fps_time = current_time

            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)

    def _zoom_camera(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        new_w = max(1, int(w / self.zoom_factor))
        new_h = max(1, int(h / self.zoom_factor))
        x_offset = min(max(0, self.x_offset), w - new_w)
        y_offset = min(max(0, self.y_offset), h - new_h)
        cropped = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def capture_image(self) -> np.ndarray:
        with self.frame_lock():
            if not self.frame_buffer:
                return np.zeros((*self.DEFAULT_CNN_SIZE, 3), dtype=np.uint8)

            frame = np.mean(self.frame_buffer, axis=0).astype(np.uint8) if len(
                self.frame_buffer) > 1 else self.frame_buffer[-1]
            return cv2.resize(frame, self.DEFAULT_CNN_SIZE, interpolation=cv2.INTER_LINEAR)

    def process_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB) if image.dtype == np.uint8 else image
        return np.multiply(image, 1.0 / 255.0, dtype=np.float32)

    def detect_objects_async(self, frame: np.ndarray):
        return self.executor.submit(self.detect_objects, frame)

    def detect_largest_object(self, frame: np.ndarray, color_range: ColorRange) -> Optional[Rectangle]:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            return Rectangle(x, y, w, h)
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return None

    def detect_objects(self, frame: np.ndarray) -> Tuple[Optional[Rectangle], Optional[Rectangle]]:
        futures = {
            'pink': self.executor.submit(self.detect_largest_object, frame, self.COLOR_RANGES['pink']),
            'blue': self.executor.submit(self.detect_largest_object, frame, self.COLOR_RANGES['blue'])
        }
        return (futures['pink'].result(), futures['blue'].result())

    def calculate_distance(self, pink_rect: Rectangle, blue_rect: Rectangle) -> float:
        return float(np.linalg.norm(np.array(pink_rect.center) - np.array(blue_rect.center))) if pink_rect and blue_rect else float('inf')

    def check_if_pink_on_blue(self, pink_rect: Optional[Rectangle], blue_rect: Optional[Rectangle]) -> bool:
        return pink_rect.overlaps(blue_rect) if pink_rect and blue_rect else False

    def release(self) -> None:
        if hasattr(self, '_running'):
            self._running.clear()
        if hasattr(self, 'capture_process'):
            self.capture_process.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def __enter__(self) -> 'CameraProcessor':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()
