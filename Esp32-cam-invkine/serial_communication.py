import serial
import struct
import argparse
import time
import random

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Serial communication')
    parser.add_argument('--port', type=str, default='COM3', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate')
    args = parser.parse_args()
    
    INPUT_SHAPE = 3
    OUTPUT_SHAPE = 2
    
    esp32cam_serial = serial.Serial(port=args.port, baudrate=args.baudrate, timeout=1.0)
    esp32cam_serial.setDTR(False)
    esp32cam_serial.setRTS(False)

    message = "1,2,3"
    esp32cam_serial.write(message.encode())
    data = esp32cam_serial.read_until("\n")
    esp32cam_serial.flush()

    try:
        while True:
            message = str(input("Enter message: "))
            esp32cam_serial.write(message.encode())
            time.sleep(0.00001)
            data = esp32cam_serial.readline()
            print(data)
            
    except KeyboardInterrupt:
        esp32cam_serial.close()
        print("Serial port closed") 