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
    v1 = np.array([p1[0] - joint[0], p1[1] - joint[1]])  # Vector p1 to joint
    v2 = np.array([p2[0] - joint[0], p2[1] - joint[1]])  # Vector p2 to joint

    # Compute the angle between vectors
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)
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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = hand_landmarks.landmark

            # Get the coordinates of key landmarks (example: index finger)
            joint = [landmarks[0].x, landmarks[0].y]  # Wrist
            p1 = [landmarks[4].x, landmarks[4].y]     # Thumb tip
            p2 = [landmarks[8].x, landmarks[8].y]     # Index finger tip

            # Calculate the angle between wrist, thumb tip, and index finger tip
            angle = calculate_angle(joint, p1, p2)

            # Display the angle on the frame
            cv2.putText(frame, str(int(angle)), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Hand Tracking', frame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
