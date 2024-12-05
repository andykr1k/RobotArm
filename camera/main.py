from flask import Flask, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

app = Flask(__name__)

model = YOLO("yolov8n.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device).eval()

midas_transforms = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

cap = cv2.VideoCapture(0)
zoom_factor = 1.0
x_offset, y_offset = 120, 40
distance_from_pink_to_blue = 0
on_blue_text = 0
left_gripper_distance = 0
right_gripper_distance = 0


def yolov8_detect(frame):
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


def estimate_depth(frame):
    """Generate a depth map from a frame using MiDaS."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    input_batch = midas_transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = midas(input_batch)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255,
                              cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)


def generate_frame_depth():
    success, frame = cap.read()
    if not success:
        return None
    frame = cv2.flip(frame, -1)
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    depth_frame = estimate_depth(frame)
    ret, buffer = cv2.imencode('.jpg', depth_frame)
    return buffer.tobytes()


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


def detect_largest_object(frame, mask, grid_color, dot_color=(0, 0, 255)):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No object detected in the specified mask.")
        return None

    largest_contour = max(contours, key=cv2.contourArea, default=None)

    if largest_contour is not None:
        area = cv2.contourArea(largest_contour)
        if area > 50:
            cv2.drawContours(frame, [largest_contour], -1, grid_color, 2)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                cv2.circle(frame, (center_x, center_y), 5, dot_color, -1)
                return (center_x, center_y)
    return None


def generate_frame_OD():
    global distance_from_pink_to_blue, on_blue_text, left_gripper_distance, right_gripper_distance

    success, frame = cap.read()
    if not success:
        print("Failed to read from the camera.")
        return None

    frame = cv2.flip(frame, -1)

    lower_green = np.array([50, 50, 50])
    upper_green = np.array([75, 255, 255])

    lower_pink = np.array([120, 50, 190])
    upper_pink = np.array([160, 90, 240])

    lower_blue = np.array([90, 130, 128])
    upper_blue = np.array([120, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)

    pink_center = detect_largest_object(frame, mask_pink, (255, 0, 255))
    blue_center = detect_largest_object(frame, mask_blue, (255, 0, 0))
    green_center = detect_largest_object(frame, mask_green, (0, 255, 0))

    distance_from_pink_to_blue, on_blue_text, right_gripper_distance, left_gripper_distance = 0, 0, 0, 0
    if pink_center and blue_center and green_center:
        distance_from_pink_to_blue = int(np.linalg.norm(
            np.array(pink_center) - np.array(blue_center)))
        right_gripper_distance = int(np.linalg.norm(
            np.array(pink_center) - np.array(green_center)))

        on_blue = (abs(pink_center[0] - blue_center[0]) <
                   38 and abs(pink_center[1] - blue_center[1]) < 38)
        on_blue_text = 1 if on_blue else 0

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        print("Failed to encode the frame.")
        return None

    frame = buffer.tobytes()
    return frame


def generate_frame_yolov8():
    success, frame = cap.read()
    frame = cv2.flip(frame, -1)
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    frame = yolov8_detect(frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame


def generate_frame_ED():
    success, frame = cap.read()
    frame = cv2.flip(frame, -1)
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    frame = multiplyFrameWithEDKernal(frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame


def generate_frame():
    global distance_from_pink_to_blue, on_blue_text, left_gripper_distance, right_gripper_distance
    success, frame = cap.read()
    frame = cv2.flip(frame, -1)
    frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    return frame


def generate_frames_OD():
    global distance_from_pink_to_blue, on_blue_text, left_gripper_distance, right_gripper_distance
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, -1)

        lower_green = np.array([50, 50, 50])
        upper_green = np.array([75, 255, 255])

        lower_pink = np.array([120, 50, 190])
        upper_pink = np.array([160, 90, 240])

        lower_blue = np.array([90, 130, 128])
        upper_blue = np.array([120, 255, 255])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)

        pink_center = detect_largest_object(frame, mask_pink, (255, 0, 255))
        blue_center = detect_largest_object(frame, mask_blue, (255, 0, 0))
        green_center = detect_largest_object(frame, mask_green, (0, 255, 0))

        distance_from_pink_to_blue, on_blue_text, right_gripper_distance, left_gripper_distance = 0, 0, 0, 0
        if pink_center and blue_center and green_center:
            distance_from_pink_to_blue = int(np.linalg.norm(
                np.array(pink_center) - np.array(blue_center)))
            right_gripper_distance = int(np.linalg.norm(
                np.array(pink_center) - np.array(green_center)))

            on_blue = (abs(pink_center[0] - blue_center[0]) <
                       38 and abs(pink_center[1] - blue_center[1]) < 38)
            on_blue_text = 1 if on_blue else 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_depth():
    while True:
        frame = generate_frame_depth()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_yolov8():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, -1)
        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        frame = yolov8_detect(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_ED():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, -1)
        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        frame = multiplyFrameWithEDKernal(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames():
    global distance_from_pink_to_blue, on_blue_text, left_gripper_distance, right_gripper_distance
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, -1)
        frame = zoom_camera(frame, zoom_factor, x_offset, y_offset)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed_depth')
def video_feed_depth():
    return Response(generate_frames_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
        "distance": distance_from_pink_to_blue,
        "on_blue": on_blue_text,
        "left_gripper_distance": left_gripper_distance,
        "right_gripper_distance": right_gripper_distance
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
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Left Gripper Distance: ${data.left_gripper_distance} Right Gripper Distance: ${data.right_gripper_distance} Pink on Blue: ${data.on_blue}`;
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
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Left Gripper Distance: ${data.left_gripper_distance} Right Gripper Distance: ${data.right_gripper_distance} Pink on Blue: ${data.on_blue}`;
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
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Left Gripper Distance: ${data.left_gripper_distance} Right Gripper Distance: ${data.right_gripper_distance} Pink on Blue: ${data.on_blue}`;
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


@app.route('/depth')
def depth():
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
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Left Gripper Distance: ${data.left_gripper_distance} Right Gripper Distance: ${data.right_gripper_distance} Pink on Blue: ${data.on_blue}`;
                    } catch (error) {
                        console.error('Error fetching info:', error);
                    }
                }
                setInterval(fetchInfo, 500);
            </script>
        </head>
        <body>
            <div id="info" style="font-size:18px; color:#333; margin-bottom:10px;"></div>
            <img src="/video_feed_depth" width="640" height="480">
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
                            `Zoom: ${data.zoom_factor.toFixed(1)}, X Offset: ${data.x_offset}, Y Offset: ${data.y_offset} Distance: ${data.distance} Left Gripper Distance: ${data.left_gripper_distance} Right Gripper Distance: ${data.right_gripper_distance} Pink on Blue: ${data.on_blue}`;
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
                <li><a href="/depth">Depth Stream</a></li>
                <li><a href="/info">Info Stream</a></li>
            </ul>
        </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
