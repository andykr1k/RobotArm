import threading
import cv2
import numpy as np


class CameraProcessor:
    def __init__(self, zoom_factor=2.2, x_offset=430, y_offset=140):
        self.cap = cv2.VideoCapture(0)
        self.zoom_factor = zoom_factor
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.running = True
        self.frame = np.zeros((128, 128, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                zoomed_frame = self.zoom_camera(frame)
                self.frame = self.process_image(zoomed_frame)

    def process_image(self, image):
        image_resized = cv2.resize(image, (128, 128)) / 255.0
        return image_resized.reshape(1, 128, 128, 3)

    def zoom_camera(self, frame):
        h, w = frame.shape[:2]
        new_w, new_h = int(w / self.zoom_factor), int(h / self.zoom_factor)
        if new_w <= 0 or new_h <= 0:
            return frame
        x_offset = max(0, min(self.x_offset, w - new_w))
        y_offset = max(0, min(self.y_offset, h - new_h))
        cropped_frame = frame[y_offset:y_offset +
                              new_h, x_offset:x_offset + new_w]
        return cv2.resize(cropped_frame, (w, h))

    def capture_image(self):
        with self.lock:
            return self.frame.copy()

    def release(self):
        self.running = False
        self.capture_thread.join()
        self.cap.release()

    def detect_largest_object(self, frame, lower_color, upper_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_contour)
        return None

    def get_distance_between_objects(self, pink_rect, blue_rect):
        if pink_rect and blue_rect:
            pink_center = (pink_rect[0] + pink_rect[2] //
                           2, pink_rect[1] + pink_rect[3] // 2)
            blue_center = (blue_rect[0] + blue_rect[2] //
                           2, blue_rect[1] + blue_rect[3] // 2)
            return int(np.linalg.norm(np.array(pink_center) - np.array(blue_center)))
        return None

    def is_pink_on_blue(self, pink_rect, blue_rect):
        if pink_rect and blue_rect:
            return (pink_rect[0] < blue_rect[0] + blue_rect[2] and
                    pink_rect[0] + pink_rect[2] > blue_rect[0] and
                    pink_rect[1] < blue_rect[1] + blue_rect[3] and
                    pink_rect[1] + pink_rect[3] > blue_rect[1])
        return False

    def detect_objects(self, frame):
        lower_pink = np.array([130, 34, 175])
        upper_pink = np.array([180, 255, 255])
        lower_blue = np.array([90, 130, 128])
        upper_blue = np.array([180, 255, 255])

        pink_rect = self.detect_largest_object(frame, lower_pink, upper_pink)
        blue_rect = self.detect_largest_object(frame, lower_blue, upper_blue)

        return pink_rect, blue_rect

    def calculate_distance(self, frame):
        pink_rect, blue_rect = self.detect_objects(frame)
        return self.get_distance_between_objects(pink_rect, blue_rect) or float('inf')

    def check_if_pink_on_blue(self, frame):
        pink_rect, blue_rect = self.detect_objects(frame)
        return self.is_pink_on_blue(pink_rect, blue_rect)

    def render_image(self, image):
        cv2.imshow('RobotArmEnv', image)
        cv2.waitKey(1)
