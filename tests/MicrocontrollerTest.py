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


send_to_esp("{7:0},{11:0},{15:0}")
send_to_esp("{7:15},{11:15},{15:15}")
send_to_esp("{7:30},{11:30},{15:30}")
send_to_esp("{7:45},{11:45},{15:45}")
send_to_esp("{7:0},{11:0},{15:0}")
