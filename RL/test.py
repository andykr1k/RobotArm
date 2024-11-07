import cv2
import multiprocessing
import random
import time


def test_camera():
    """Function to capture and display video from the camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow("Test Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def math_operations():
    """Function to perform random math operations indefinitely."""
    while True:
        num1 = random.randint(1, 100)
        num2 = random.randint(1, 100)

        # Random math operation
        result_add = num1 + num2
        result_sub = num1 - num2
        result_mul = num1 * num2
        # Avoid division by zero
        result_div = num1 / (num2 if num2 != 0 else 1)

        # Print out the results for demonstration
        print(f"Addition: {num1} + {num2} = {result_add}")
        print(f"Subtraction: {num1} - {num2} = {result_sub}")
        print(f"Multiplication: {num1} * {num2} = {result_mul}")
        print(f"Division: {num1} / {num2} = {result_div:.2f}")

        # Sleep briefly to slow down the output
        time.sleep(1)


if __name__ == "__main__":
    # Create a separate process for the math operations
    math_process = multiprocessing.Process(target=math_operations)

    # Start the math process
    math_process.start()

    # Run the camera function in the main process
    test_camera()

    # Terminate the math process after the camera feed is closed
    math_process.terminate()
    math_process.join()
