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


def send_command_to_esp(motor_angle_pairs):
    command_parts = []

    for motor, angle in motor_angle_pairs:
        if angle is not None:
            command_parts.append(f'{motor}:{180 - int(angle)}')

    command = '{' + '},{'.join(command_parts) + '}\n'

    try:
        ser.write(command.encode())
        print(f'Sent command: {command.strip()}')
    except serial.SerialException as e:
        print(f"Error sending command: {e}")


def motor_control_worker(angles_queue):
    last_angles = {15: None, 7: None, 11: None, 0: None}

    while True:
        if not angles_queue.empty():
            # tip_motor, tip_angle, mid_motor, mid_angle, base_motor, base_angle = angles_queue.get()

            wrist_motor, wrist_rotate, tip_motor, tip_angle, mid_motor, mid_angle, base_moter, base_angle = angles_queue.get()

            # last_angle = last_angles.get(tip_motor)
            # if last_angle is None or abs(tip_angle - last_angle) > 5:
            #     last_angles[tip_motor] = tip_angle
            # else:
            #     print(
            #         f'Angle change for motor {tip_motor} is less than 5 degrees; skipping command.')

            # last_angle = last_angles.get(mid_motor)
            # if last_angle is None or abs(mid_angle - last_angle) > 5:
            #     last_angles[mid_motor] = mid_angle
            # else:
            #     print(
            #         f'Angle change for motor {mid_motor} is less than 5 degrees; skipping command.')

            # last_angle = last_angles.get(base_motor)
            # if last_angle is None or abs(base_angle - last_angle) > 5:
            #     last_angles[base_motor] = base_angle
            # else:
            #     print(
            #         f'Angle change for motor {base_motor} is less than 5 degrees; skipping command.')

            # send_command_to_esp(
            #     [(tip_motor, last_angles[tip_motor]), (mid_motor, last_angles[mid_motor]), (base_motor, last_angles[base_motor])])
            
            send_command_to_esp(
                [(wrist_motor, 90 + wrist_rotate), (mid_motor, mid_angle), (tip_motor, tip_angle), (base_moter, base_angle)])


def hand_tracking(angles_queue):
    send_command_to_esp([(0, 90), (4, 180), (11, 180), (15, 180)])
    cap = cv2.VideoCapture(0)
    frames = 0
    wrist_rotation = 0
    while cap.isOpened():
        frames += 1
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
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

                for i in range(len(points) - 2):
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

                # Display angles on the frame next to each point
                cv2.putText(frame, f'{int(angle_top)} deg', (points[2][0] - 70, points[2][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f'{int(angle_middle)} deg', (points[1][0] - 70, points[1][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f'{int(angle_base)} deg', (points[0][0] - 70, points[0][1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if (frames % 10):
                    # angles_queue.put(
                    #     (15, angle_top, 11, angle_middle, 7, angle_base))
                    angles_queue.put(
                        (0, wrist_rotation, 4, angle_top, 11, angle_middle, 15, angle_base))

            box_color = (0, 255, 0) if hand_detected else (
                0, 0, 255)
            cv2.rectangle(frame, (10, 10), (160, 60),
                          box_color, -1)
            cv2.putText(frame, 'Hand Detected' if hand_detected else 'No Hand', (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Index Finger Tracking', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            wrist_rotation -= 3
        elif key == ord('d'):
            wrist_rotation += 3
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    angles_queue = queue.Queue()

    motor_thread = threading.Thread(
        target=motor_control_worker, args=(angles_queue,))
    motor_thread.daemon = True
    motor_thread.start()

    hand_tracking(angles_queue)
