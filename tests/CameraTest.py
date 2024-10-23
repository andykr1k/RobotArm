import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)  # Wait for a key press
else:
    print("Failed to capture image")

cap.release()
cv2.destroyAllWindows()
