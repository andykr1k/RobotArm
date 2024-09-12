import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points (joint, p1, p2)
def calculate_angle(joint, p1, p2):
    # Vector from joint to p1
    v1 = np.array([p1[0] - joint[0], p1[1] - joint[1]])
    # Vector from joint to p2
    v2 = np.array([p2[0] - joint[0], p2[1] - joint[1]])

    # Normalize vectors and compute cosine of the angle
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Ensure valid input for arccos
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # Check if any hand is detected and set the status color
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True  # Hand is detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of index finger landmarks
            index_mcp = [hand_landmarks.landmark[5].x,
                         hand_landmarks.landmark[5].y]  # MCP
            index_pip = [hand_landmarks.landmark[6].x,
                         hand_landmarks.landmark[6].y]  # PIP
            index_dip = [hand_landmarks.landmark[7].x,
                         hand_landmarks.landmark[7].y]  # DIP
            index_tip = [hand_landmarks.landmark[8].x,
                         hand_landmarks.landmark[8].y]  # Tip
            wrist = [hand_landmarks.landmark[0].x,
                     hand_landmarks.landmark[0].y]  # Wrist landmark

            # Convert normalized coordinates to pixel values
            height, width, _ = frame.shape
            points = [index_mcp, index_pip, index_dip, index_tip, wrist]
            points = [(int(point[0] * width), int(point[1] * height))
                      for point in points]

            # Names of points for display
            point_names = ["MCP", "PIP", "DIP", "Tip", "Wrist"]

            # Draw red lines connecting index finger joints
            for i in range(len(points) - 2):  # Skip wrist connection for now
                cv2.line(frame, points[i], points[i + 1],
                         (0, 0, 255), 2)  # Red line

            # Draw circles on joints and display point names
            for i, point in enumerate(points):
                cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Green circles
                cv2.putText(frame, point_names[i], (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Calculate angles and display them on the frame
            # Angle 1: Top Finger (Tip -> DIP -> PIP)
            angle_top = calculate_angle(
                points[2], points[1], points[3])  # DIP joint angle
            # Angle 2: Middle Finger (DIP -> PIP -> MCP)
            angle_middle = calculate_angle(
                points[1], points[0], points[2])  # PIP joint angle
            # Angle 3: Finger Base (PIP -> MCP -> Wrist)
            angle_base = calculate_angle(
                points[0], points[1], points[4])  # MCP joint angle

            # Display angles on the frame next to each point
            cv2.putText(frame, f'{int(angle_top)} deg', (points[2][0] - 70, points[2][1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f'{int(angle_middle)} deg', (points[1][0] - 70, points[1][1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f'{int(angle_base)} deg', (points[0][0] - 70, points[0][1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw the status box at the top left
    box_color = (0, 255, 0) if hand_detected else (
        0, 0, 255)  # Green if hand is detected, red otherwise
    cv2.rectangle(frame, (10, 10), (160, 60), box_color, -1)  # Status box
    cv2.putText(frame, 'Hand Detected' if hand_detected else 'No Hand', (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Index Finger Tracking', frame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
