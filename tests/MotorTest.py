import time
import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

# Initialize I2C interface and PCA9685 instance
i2c = board.I2C()  # uses board.SCL and board.SDA
pca = PCA9685(i2c)
pca.frequency = 50  # Set frequency to 50Hz for servos

# Create servo objects for channels 4, 5, and 6
servo4 = servo.Servo(pca.channels[4])
servo5 = servo.Servo(pca.channels[5])
servo6 = servo.Servo(pca.channels[6])

# Move all servos to 0 degrees (their home position)
servo4.angle = 0
servo5.angle = 0
servo6.angle = 0

# Give the servos time to move to the 0-degree position
time.sleep(1)  # Adjust the time if needed based on your servos

# Move all servos to 0 degrees (their home position)
servo4.angle = 180
servo5.angle = 180
servo6.angle = 180

# Give the servos time to move to the 0-degree position
time.sleep(1)  # Adjust the time if needed based on your servos

# Move all servos to 0 degrees (their home position)
servo4.angle = 0
servo5.angle = 0
servo6.angle = 0

# Give the servos time to move to the 0-degree position
time.sleep(1)  # Adjust the time if needed based on your servos

# Stop the PCA9685 (optional cleanup)
pca.deinit()
