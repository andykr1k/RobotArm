import serial
import time

ser = serial.Serial('/dev/cu.usbserial-0001', 9600, timeout=1)
time.sleep(2)

def read_from_esp():
    """Read and print data from the ESP."""
    while ser.in_waiting > 0:
        line = ser.readline().rstrip()
        if line:
            print(f"Received from ESP: {line}")

def send_to_esp(command):
    print(command)
    ser.write(command.encode())
    time.sleep(4)
    read_from_esp()

send_to_esp("{4:0},{5:0}")
send_to_esp("{4:45},{5:45}")
send_to_esp("{4:0},{5:0}")
