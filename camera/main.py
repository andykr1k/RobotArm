from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)

cap = cv2.VideoCapture(0)
zoom_factor = 2.2
x_offset, y_offset = 430, 140

def zoom_camera(frame, zoom_factor, x_offset, y_offset):
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    if new_w <= 0 or new_h <= 0:
        return frame
    x_offset = max(0, min(x_offset, w - new_w))
    y_offset = max(0, min(y_offset, h - new_h))
    cropped_frame = frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    return cv2.resize(cropped_frame, (w, h))

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

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        # Define color ranges for detection
        lower_pink = np.array([130, 34, 175])
        upper_pink = np.array([180, 255, 255])
        lower_blue = np.array([90, 130, 128])
        upper_blue = np.array([180, 255, 255])

        # Apply zoom and detect objects
        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        pink_rect = detect_largest_object(
            frame, lower_pink, upper_pink, (255, 0, 0))
        blue_rect = detect_largest_object(
            frame, lower_blue, upper_blue, (0, 255, 0))

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return frame in an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Object Detection Stream</title>
        </head>
        <body>
            <h1>Object Detection Stream</h1>
            <img src="/video_feed">
        </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
