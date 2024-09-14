import time
import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

# Initialize I2C interface and PCA9685 instance
i2c = board.I2C()  # uses board.SCL and board.SDA
pca = PCA9685(i2c)
pca.frequency = 50  # Set frequency to 50Hz for servos

# Set min_pulse and max_pulse for 180-degree movement
min_pulse = 500   # ~1ms pulse width (0 degrees)
max_pulse = 2400  # ~2.4ms pulse width (180 degrees)

# Create servo objects for channels 4, 5, and 6 with the adjusted pulse range
servo4 = servo.Servo(pca.channels[4], min_pulse=min_pulse, max_pulse=max_pulse)
servo5 = servo.Servo(pca.channels[5], min_pulse=min_pulse, max_pulse=max_pulse)

# Function to move servos to a given angle


def move_servos(angle, delay=1):
    servo4.angle = angle
    servo5.angle = angle
    time.sleep(delay)  # Wait for the servos to move to the position


# Move servos to 0 degrees (home position)
move_servos(0)

# Move servos to 180 degrees
move_servos(90)

# Move servos back to 0 degrees
move_servos(0)

# Stop the PCA9685 (optional cleanup)
pca.deinit()
