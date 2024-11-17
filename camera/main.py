from flask import Flask, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
zoom_factor = 1.7
x_offset, y_offset = 100, 60
distance_text = 0
on_blue_text = 0


def yolov8(frame):
    results = model.predict(frame)
    annotated_frame = frame.copy()
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{model.names[int(class_id)]}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    return annotated_frame


def multiplyFrameWithEDKernal(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)

    edges = cv2.magnitude(edges_x, edges_y)
    edges = np.clip(edges, 0, 255).astype(np.uint8)

    result_frame = cv2.merge([edges, edges, edges])

    return result_frame


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


def generate_frame_OD():
    global distance_text, on_blue_text
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    lower_pink = np.array([130, 34, 175])
    upper_pink = np.array([180, 255, 255])
    lower_blue = np.array([90, 130, 128])
    upper_blue = np.array([180, 255, 255])

    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    pink_rect = detect_largest_object(
        frame, lower_pink, upper_pink, (255, 0, 0))
    blue_rect = detect_largest_object(
        frame, lower_blue, upper_blue, (0, 255, 0))

    distance_text, on_blue_text = 0, 0
    if pink_rect and blue_rect:
        pink_center = (pink_rect[0] + pink_rect[2] //
                       2, pink_rect[1] + pink_rect[3] // 2)
        blue_center = (blue_rect[0] + blue_rect[2] //
                       2, blue_rect[1] + blue_rect[3] // 2)
        distance_text = int(np.linalg.norm(
            np.array(pink_center) - np.array(blue_center)))
        on_blue = (pink_rect[0] < blue_rect[0] + blue_rect[2] and
                   pink_rect[0] + pink_rect[2] > blue_rect[0] and
                   pink_rect[1] < blue_rect[1] + blue_rect[3] and
                   pink_rect[1] + pink_rect[3] > blue_rect[1])
        on_blue_text = 1 if on_blue else 0

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()

    return frame


def generate_frame_yolov8():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    frame = yolov8(frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame


def generate_frame_ED():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    frame = multiplyFrameWithEDKernal(frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame


def generate_frame():
    global distance_text, on_blue_text
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame


def generate_frames_OD():
    global distance_text, on_blue_text
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        lower_pink = np.array([130, 34, 175])
        upper_pink = np.array([180, 255, 255])
        lower_blue = np.array([90, 130, 128])
        upper_blue = np.array([180, 255, 255])

        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        pink_rect = detect_largest_object(
            frame, lower_pink, upper_pink, (255, 0, 0))
        blue_rect = detect_largest_object(
            frame, lower_blue, upper_blue, (0, 255, 0))

        distance_text, on_blue_text = 0, 0
        if pink_rect and blue_rect:
            pink_center = (pink_rect[0] + pink_rect[2] //
                           2, pink_rect[1] + pink_rect[3] // 2)
            blue_center = (blue_rect[0] + blue_rect[2] //
                           2, blue_rect[1] + blue_rect[3] // 2)
            distance_text = int(np.linalg.norm(
                np.array(pink_center) - np.array(blue_center)))
            on_blue = (pink_rect[0] < blue_rect[0] + blue_rect[2] and
                       pink_rect[0] + pink_rect[2] > blue_rect[0] and
                       pink_rect[1] < blue_rect[1] + blue_rect[3] and
                       pink_rect[1] + pink_rect[3] > blue_rect[1])
            on_blue_text = 1 if on_blue else 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_yolov8():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        frame = yolov8(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_ED():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        frame = multiplyFrameWithEDKernal(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames():
    global distance_text, on_blue_text
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/pic_feed_yolov8')
def pic_feed_yolov8():
    return Response(generate_frame_yolov8(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pic_feed_ED')
def pic_feed_ED():
    return Response(generate_frame_ED(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pic_feed_OD')
def pic_feed_OD():
    return Response(generate_frame_OD(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pic_feed')
def pic_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_yolov8')
def video_feed_yolov8():
    return Response(generate_frames_yolov8(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_ED')
def video_feed_ED():
    return Response(generate_frames_ED(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_OD')
def video_feed_OD():
    return Response(generate_frames_OD(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/info')
def info():
    return jsonify({
        "zoom_factor": zoom_factor,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "distance": distance_text,
        "on_blue": on_blue_text
    })


@app.route('/live')
def live():
    return '''
    <html>
        <head>
            <title>Robot Arm Camera Stream</title>
            <script>
                async function fetchInfo() {
                    try {
                        const response = await fetch('/info');
                        const data = await response.json();
                        document.getElementById('info').innerText = 
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Pink on Blue: ${data.on_blue}`;
                    } catch (error) {
                        console.error('Error fetching info:', error);
                    }
                }
                setInterval(fetchInfo, 500);
            </script>
        </head>
        <body>
            <div id="info" style="font-size:18px; color:#333; margin-bottom:10px;"></div>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    '''


@app.route('/OD')
def OD():
    return '''
    <html>
        <head>
            <title>Robot Arm Camera Stream</title>
            <script>
                async function fetchInfo() {
                    try {
                        const response = await fetch('/info');
                        const data = await response.json();
                        document.getElementById('info').innerText = 
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Pink on Blue: ${data.on_blue}`;
                    } catch (error) {
                        console.error('Error fetching info:', error);
                    }
                }
                setInterval(fetchInfo, 500);
            </script>
        </head>
        <body>
            <div id="info" style="font-size:18px; color:#333; margin-bottom:10px;"></div>
            <img src="/video_feed_OD" width="640" height="480">
        </body>
    </html>
    '''


@app.route('/ED')
def ED():
    return '''
    <html>
        <head>
            <title>Robot Arm Camera Stream</title>
            <script>
                async function fetchInfo() {
                    try {
                        const response = await fetch('/info');
                        const data = await response.json();
                        document.getElementById('info').innerText = 
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Pink on Blue: ${data.on_blue}`;
                    } catch (error) {
                        console.error('Error fetching info:', error);
                    }
                }
                setInterval(fetchInfo, 500);
            </script>
        </head>
        <body>
            <div id="info" style="font-size:18px; color:#333; margin-bottom:10px;"></div>
            <img src="/video_feed_ED" width="640" height="480">
        </body>
    </html>
    '''


@app.route('/yolov8')
def yolov8():
    return '''
    <html>
        <head>
            <title>Robot Arm Camera Stream</title>
            <script>
                async function fetchInfo() {
                    try {
                        const response = await fetch('/info');
                        const data = await response.json();
                        document.getElementById('info').innerText = 
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Pink on Blue: ${data.on_blue}`;
                    } catch (error) {
                        console.error('Error fetching info:', error);
                    }
                }
                setInterval(fetchInfo, 500);
            </script>
        </head>
        <body>
            <div id="info" style="font-size:18px; color:#333; margin-bottom:10px;"></div>
            <img src="/video_feed_yolov8" width="640" height="480">
        </body>
    </html>
    '''


@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Robot Arm Camera Stream</title>
        </head>
        <body>
            <h1>Robot Arm Camera Stream</h1>
            <ul>
                <li><a href="/live">Live Stream</a></li>
                <li><a href="/OD">Object Detection Stream</a></li>
                <li><a href="/ED">Edge Detection Stream</a></li>
                <li><a href="/yolov8">Smart Object Detection Stream</a></li>
                <li><a href="/info">Info Stream</a></li>
            </ul>
        </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
