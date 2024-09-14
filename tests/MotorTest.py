import time
import board
from adafruit_pca9685 import PCA9685

# Create the I2C bus interface.
i2c = board.I2C()  # uses board.SCL and board.SDA

# Create a simple PCA9685 class instance.
pca = PCA9685(i2c)

# Set the PWM frequency to 60hz for servos.
pca.frequency = 60

# Define the channel for the servo (4th slot = index 3)
servo_channel = pca.channels[3]

# Define min and max duty cycles for 0 and 180 degrees
min_pulse = 0x199A  # ~1ms pulse for 0 degrees
max_pulse = 0x3333  # ~2ms pulse for 180 degrees

# Function to move servo to a specific position


def move_servo(position):
    if position == 0:
        servo_channel.duty_cycle = min_pulse
    elif position == 180:
        servo_channel.duty_cycle = max_pulse


# Move servo back and forth for 5 seconds
start_time = time.time()
while time.time() - start_time < 5:
    # Move to 0 degrees
    move_servo(0)
    time.sleep(1)  # Hold for 1 second

    # Move to 180 degrees
    move_servo(180)
    time.sleep(1)  # Hold for 1 second

# After 5 seconds, stop the servo by setting the duty cycle to 0
servo_channel.duty_cycle = 0

# Turn off the PCA9685
pca.deinit()
