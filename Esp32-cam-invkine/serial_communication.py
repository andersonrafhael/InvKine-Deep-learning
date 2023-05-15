import serial
import struct
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Serial communication')
    parser.add_argument('--port', type=str, default='COM5', help='Serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate')
    args = parser.parse_args()
    
    serial_conn = serial.Serial()
    serial_conn.baudrate = args.baudrate
    serial_conn.port = args.port
    serial_conn.open()
    
    while True:
        
        # Perform computation using coordinates received from microcontroller
        x, y, z = 1.0, 2.0, 3.0
        coords = struct.pack('fff', x, y, z)
        
        # Pack the result as a float and send it to the microcontroller
        serial_conn.write(coords)
        
        # Wait for and receive data from the microcontroller
        response_bytes = serial_conn.readline()
        
        # Decode the received data and extract any values sent by the microcontroller
        if response_bytes:
            response_values = struct.unpack('ff', response_bytes)
            print(response_values)
    
    
    serial_conn.close()
