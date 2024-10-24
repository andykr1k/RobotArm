import threading
import cv2
import numpy as np
import time
from typing import Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ColorRange:
    """Color range definition in HSV space"""
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class Rectangle:
    """Rectangle definition with x, y, width, height"""
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate center point of rectangle"""
        return (self.x + self.width // 2, self.y + self.height // 2)


class CameraProcessor:
    """
    Handles camera capture and image processing for robot arm control
    """
    COLOR_RANGES = {
        'pink': ColorRange(
            lower=np.array([130, 34, 175]),
            upper=np.array([180, 255, 255])
        ),
        'blue': ColorRange(
            lower=np.array([90, 130, 128]),
            upper=np.array([180, 255, 255])
        )
    }

    def __init__(self, zoom_factor: float = 2.2, x_offset: int = 430, y_offset: int = 140, camera_id: int = 0):
        self.zoom_factor = zoom_factor
        self.x_offset = x_offset
        self.y_offset = y_offset

        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")

        # Initialize state
        self.running = True
        self.frame = None
        self.lock = threading.Lock()

        # Thread pool for processing tasks
        self.executor = ThreadPoolExecutor(max_workers=4)  # Using 4 threads

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def _capture_loop(self) -> None:
        """Continuous camera capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    with self.lock:
                        self.frame = self.zoom_camera(frame)
                except Exception as e:
                    print(f"Error in capture loop: {e}")
            time.sleep(0.01)

    def zoom_camera(self, frame: np.ndarray) -> np.ndarray:
        """Apply zoom and offset to camera frame"""
        h, w = frame.shape[:2]
        new_w = int(w / self.zoom_factor)
        new_h = int(h / self.zoom_factor)

        if new_w <= 0 or new_h <= 0:
            return frame

        x_offset = max(0, min(self.x_offset, w - new_w))
        y_offset = max(0, min(self.y_offset, h - new_h))

        cropped = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
        return cv2.resize(cropped, (w, h))

    def capture_image(self) -> np.ndarray:
        """Thread-safe frame capture"""
        with self.lock:
            if self.frame is None:
                return np.zeros((128, 128, 3), dtype=np.uint8)
            return cv2.resize(self.frame.copy(), (128, 128))

    def process_image(self, image):
        """Process image for CNN"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        return image

    def detect_objects_async(self, frame: np.ndarray):
        """Detect pink and blue objects asynchronously"""
        return self.executor.submit(self.detect_objects, frame)

    def detect_largest_object(self, frame: np.ndarray, color_range: ColorRange) -> Optional[Rectangle]:
        """Detect largest object of specified color"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, color_range.lower, color_range.upper)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            return Rectangle(x, y, w, h)
        except Exception as e:
            print(f"Error in object detection: {e}")
            return None

    def detect_objects(self, frame: np.ndarray) -> Tuple[Optional[Rectangle], Optional[Rectangle]]:
        """Detect pink and blue objects in frame"""
        pink_rect = self.detect_largest_object(
            frame, self.COLOR_RANGES['pink'])
        blue_rect = self.detect_largest_object(
            frame, self.COLOR_RANGES['blue'])
        return pink_rect, blue_rect

    def calculate_distance(self, frame: np.ndarray) -> float:
        """Calculate distance between pink and blue objects"""
        pink_rect, blue_rect = self.detect_objects(frame)
        if pink_rect and blue_rect:
            pink_center = np.array(pink_rect.center)
            blue_center = np.array(blue_rect.center)
            return float(np.linalg.norm(pink_center - blue_center))
        return float('inf')

    def release(self) -> None:
        """Release camera resources and shutdown executor"""
        self.running = False

        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()

        # Shutdown ThreadPoolExecutor
        self.executor.shutdown(wait=True)

    def __del__(self):
        self.release()
