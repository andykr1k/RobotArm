import cv2
import numpy as np


def zoom_camera(frame, zoom_factor, x_offset, y_offset):
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    if new_w <= 0 or new_h <= 0:
        return frame
    x_offset = max(0, min(x_offset, w - new_w))
    y_offset = max(0, min(y_offset, h - new_h))
    cropped_frame = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    zoomed_frame = cv2.resize(cropped_frame, (w, h))
    return zoomed_frame


def detect_largest_object(frame, lower_color, upper_color, grid_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), grid_color, 2)
        return (x, y, w, h)
    return None


def display_info(frame, zoom_factor, x_offset, y_offset, distance_text, on_blue_text):
    info_text = f"Zoom: {zoom_factor:.1f} | X Offset: {x_offset} | Y Offset: {y_offset}"
    text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = frame.shape[1] - text_size[0] - 10
    text_y = 30
    cv2.putText(frame, info_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, distance_text, (text_x, text_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, on_blue_text, (text_x, text_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


cap = cv2.VideoCapture(0)
zoom_factor = 2.2
x_offset, y_offset = 430, 140

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    lower_pink = np.array([130, 34, 175])
    upper_pink = np.array([180, 255, 255])
    lower_blue = np.array([90, 130, 128])
    upper_blue = np.array([180, 255, 255])
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    pink_rect = detect_largest_object(
        frame, lower_pink, upper_pink, grid_color=(255, 0, 0))
    blue_rect = detect_largest_object(
        frame, lower_blue, upper_blue, grid_color=(0, 255, 0))
    distance_text = ""
    on_blue_text = ""
    if pink_rect and blue_rect:
        pink_center = (pink_rect[0] + pink_rect[2] //
                       2, pink_rect[1] + pink_rect[3] // 2)
        blue_center = (blue_rect[0] + blue_rect[2] //
                       2, blue_rect[1] + blue_rect[3] // 2)
        distance = int(np.linalg.norm(
            np.array(pink_center) - np.array(blue_center)))
        distance_text = f"Distance: {distance}"
        on_blue = (pink_rect[0] < blue_rect[0] + blue_rect[2] and
                   pink_rect[0] + pink_rect[2] > blue_rect[0] and
                   pink_rect[1] < blue_rect[1] + blue_rect[3] and
                   pink_rect[1] + pink_rect[3] > blue_rect[1])
        on_blue_text = "Pink on Blue" if on_blue else "Pink not on Blue"
    display_info(frame, zoom_factor, x_offset,
                 y_offset, distance_text, on_blue_text)
    cv2.imshow('Object Tracking', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        zoom_factor += 0.1
    elif key == ord('-'):
        if zoom_factor != 1.0:
            zoom_factor -= 0.1
    elif key == ord('w'):
        y_offset = max(0, y_offset - 10)
    elif key == ord('s'):
        y_offset += 10
    elif key == ord('a'):
        x_offset = max(0, x_offset - 10)
    elif key == ord('d'):
        x_offset += 10

cap.release()
cv2.destroyAllWindows()
