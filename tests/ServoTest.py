import serial
import time

ser = serial.Serial('/dev/cu.usbserial-0001', 9600, timeout=1)

time.sleep(2)

def read_from_esp():
    while ser.in_waiting > 0:
        line = ser.readline().rstrip()
        if line:
            print(f"Received from ESP: {line}")

def send_to_esp(command):
    try:
        print(f"Sending to ESP: {command}")
        ser.write(command.encode())
        time.sleep(4)
        read_from_esp()
    except Exception as e:
        print(f"An error occurred: {e}")

def close_connection():
    ser.close()

command = (
    '{' +
    f'0:{45},' +
    f'4:{45},' +
    f'7:{45},' +
    f'8:{45},' +
    f'11:{45},' +
    f'15:{45}' +
    '}'
)

send_to_esp(command)