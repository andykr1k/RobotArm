import cv2
import numpy as np


def zoom_and_offset(frame, zoom_factor, x_offset, y_offset):
    h, w = frame.shape[:2]

    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    x1 = int(max(0, x_offset))
    y1 = int(max(0, y_offset))
    x2 = int(min(w, x1 + new_w))
    y2 = int(min(h, y1 + new_h))

    cropped_frame = frame[y1:y2, x1:x2]
    if cropped_frame.size == 0:
        return frame

    zoomed_frame = cv2.resize(cropped_frame, (w, h))
    return zoomed_frame


def mask_colors(frame, pink_lower, pink_upper, blue_lower, blue_upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    combined_mask = cv2.bitwise_or(pink_mask, blue_mask)

    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    return masked_frame


cap = cv2.VideoCapture(0)

lower_pink = np.array([130, 34, 175])
upper_pink = np.array([180, 255, 255])

lower_blue = np.array([90, 130, 128])
upper_blue = np.array([180, 255, 255])

zoom_factor = 2.2
x_offset = 430
y_offset = 140

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame = zoom_and_offset(frame, zoom_factor, x_offset, y_offset)

    masked_frame = mask_colors(
        frame, lower_pink, upper_pink, lower_blue, upper_blue)

    # Display the masked frame
    cv2.imshow('Masked Colors', masked_frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
