import cv2
import mediapipe as mp
import numpy as np
import serial
import threading
import queue
import time

try:
    ser = serial.Serial('/dev/cu.usbserial-0001', 9600, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


def read_from_esp():
    """Read and print data from the ESP."""
    while ser.in_waiting > 0:
        line = ser.readline().rstrip()
        if line:
            print(f"Received from ESP: {line}")

def calculate_angle(joint, p1, p2):
    v1 = np.array([p1[0] - joint[0], p1[1] - joint[1]])
    v2 = np.array([p2[0] - joint[0], p2[1] - joint[1]])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def send_command_to_esp(motor1, angle1, motor2, angle2):
    command_parts = []
    if angle1 is not None:
        command_parts.append(f'{motor1}:{180 - int(angle1)}')
    if angle2 is not None:
        command_parts.append(f'{motor2}:{180 - int(angle2)}')
    command = '{' + '},{'.join(command_parts) + '}\n'

    try:
        ser.write(command.encode())
        print(f'Sent command: {command.strip()}')
    except serial.SerialException as e:
        print(f"Error sending command: {e}")


def motor_control_worker(angles_queue):
    last_angles = {4: None, 5: None}

    while True:
        if not angles_queue.empty():
            motor1, angle1, motor2, angle2 = angles_queue.get()

            last_angle1 = last_angles.get(motor1)
            if last_angle1 is None or abs(angle1 - last_angle1) > 5:
                last_angles[motor1] = angle1
            else:
                print(
                    f'Angle change for motor {motor1} is less than 5 degrees; skipping command.')

            last_angle2 = last_angles.get(motor2)
            if last_angle2 is None or abs(angle2 - last_angle2) > 5:
                last_angles[motor2] = angle2
            else:
                print(
                    f'Angle change for motor {motor2} is less than 5 degrees; skipping command.')

            send_command_to_esp(
                motor1, last_angles[motor1], motor2, last_angles[motor2])

            time.sleep(1)

            read_from_esp()


def hand_tracking(angles_queue):
    cap = cv2.VideoCapture(0)
    frame_rate = 10
    frame_delay = 1.0 / frame_rate

    prev_time = time.time()

    while cap.isOpened():
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time >= frame_delay:
            prev_time = current_time

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_mcp = [hand_landmarks.landmark[5].x,
                                 hand_landmarks.landmark[5].y]
                    index_pip = [hand_landmarks.landmark[6].x,
                                 hand_landmarks.landmark[6].y]
                    index_dip = [hand_landmarks.landmark[7].x,
                                 hand_landmarks.landmark[7].y]
                    index_tip = [hand_landmarks.landmark[8].x,
                                 hand_landmarks.landmark[8].y]
                    wrist = [hand_landmarks.landmark[0].x,
                             hand_landmarks.landmark[0].y]

                    height, width, _ = frame.shape
                    points = [index_mcp, index_pip,
                              index_dip, index_tip, wrist]
                    points = [(int(point[0] * width), int(point[1] * height))
                              for point in points]

                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i],
                                 points[i + 1], (0, 0, 255), 2)

                    for point in points:
                        cv2.circle(frame, point, 5, (0, 255, 0), -1)

                    angle_top = calculate_angle(
                        points[2], points[1], points[3])
                    angle_middle = calculate_angle(
                        points[1], points[0], points[2])
                    angle_base = calculate_angle(
                        points[0], points[1], points[4])

                    cv2.putText(frame, f'{int(angle_top)} deg', (points[2][0] - 70, points[2][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f'{int(angle_middle)} deg', (points[1][0] - 70, points[1][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, f'{int(angle_base)} deg', (points[0][0] - 70, points[0][1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    print(
                        f'Calculated angles - Top: {angle_top}, Middle: {angle_middle}, Base: {angle_base}')

                    angles_queue.put((5, angle_top, 4, angle_middle))

            cv2.imshow('Index Finger Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    send_command_to_esp(4, 180, 5, 180)


if __name__ == "__main__":
    angles_queue = queue.Queue()

    motor_thread = threading.Thread(
        target=motor_control_worker, args=(angles_queue,))
    motor_thread.daemon = True
    motor_thread.start()

    hand_tracking(angles_queue)
