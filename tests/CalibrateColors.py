import cv2
import numpy as np


cap = cv2.VideoCapture(0)


def nothing(x):
    pass

cv2.namedWindow('result')

h, s, v = 100, 100, 100

# Creating track bar
cv2.createTrackbar('h', 'result', 0, 255, nothing)
cv2.createTrackbar('s', 'result', 0, 255, nothing)
cv2.createTrackbar('v', 'result', 0, 255, nothing)
cv2.createTrackbar('he', 'result', 0, 255, nothing)
cv2.createTrackbar('se', 'result', 0, 255, nothing)
cv2.createTrackbar('ve', 'result', 0, 255, nothing)

zoom_factor = 1.0
x_offset, y_offset = 0, 0

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

while (1):

    _, frame = cap.read()

    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = cv2.getTrackbarPos('h', 'result')
    s = cv2.getTrackbarPos('s', 'result')
    v = cv2.getTrackbarPos('v', 'result')

    he = cv2.getTrackbarPos('he', 'result')
    se = cv2.getTrackbarPos('se', 'result')
    ve = cv2.getTrackbarPos('ve', 'result')

    lower_blue = np.array([h, s, v])
    upper_blue = np.array([he, se, ve])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('result', result)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+'):
        zoom_factor += 0.1
    elif key == ord('-'):
        if zoom_factor != 1.0:
            zoom_factor = zoom_factor - 0.1
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
