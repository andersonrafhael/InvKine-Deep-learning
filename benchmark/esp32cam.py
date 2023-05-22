import time
import serial
import argparse
import pandas as pd
from tqdm import tqdm

def warmup_esp32cam_serial(esp32cam_serial: serial.Serial) -> None:
    
    """
    This function sends a message to the ESP32-CAM to warmup the serial communication
    clear the buffer and flush the serial port.
    
    Params:
    -------
    esp32cam_serial: serial.Serial
        serial object to communicate with the ESP32-CAM
    """
    
    message = "1,2,3"
    esp32cam_serial.write(message.encode())
    _ = esp32cam_serial.read_until("\n")
    esp32cam_serial.flush()


def get_esp32_prediction(pose_str: str, esp32cam_serial: serial.Serial) -> tuple[float, list[float]]:
    
    """
    This function sends a message to the ESP32-CAM to get the prediction of the pose.
    
    Params:
    -------
    pose_str: str
        string with the pose to be predicted
    esp32cam_serial: serial.Serial
        serial object to communicate with the ESP32-CAM
    
    Returns:
    --------
    prediction: tuple[float, list[float]]
        tuple with the prediction time and the predicted theta values
    """
    
    sleep_time = 0.00001
    
    try:
        begin_time = time.time()
        esp32cam_serial.write(pose_str.encode())
        time.sleep(sleep_time)
        data = esp32cam_serial.readline().decode()
        end_time = time.time() - begin_time - sleep_time
    except:
        esp32cam_serial.close()
        raise Exception("Error in serial communication, closing serial serial communication...")
        
    thetas = [float(theta) for theta in data.split(",")]
    return end_time, thetas
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Esp32 Inference Benchmark")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment id")
    parser.add_argument("--timeout", type=float, default=1.0, help="Timeout to serial communication")
    args = parser.parse_args()

    INPUT_SHAPE = 3
    OUTPUT_SHAPE = 2
    
    test_data = pd.read_csv(f"runs/exp{args.exp_id}/test.csv")

    esp32cam_serial = serial.Serial(port=args.port, baudrate=args.baudrate, timeout=args.timeout)
    esp32cam_serial.setDTR(False)
    esp32cam_serial.setRTS(False)
    
    warmup_esp32cam_serial(esp32cam_serial)
    
    bench_data = []    
    for i, row in tqdm(test_data.iloc[:10, :].iterrows(), total=test_data.shape[0], desc="Running benchmark..."):
        
        pose_str = f"{row['x']},{row['y']},{row['z']}"
        time_pred, thetas_pred = get_esp32_prediction(pose_str, esp32cam_serial)
        bench_data.append([row.x, row.y, row.z, row.theta0, row.theta1, thetas_pred[0], thetas_pred[1], time_pred])
        
    
    bench_csv = pd.DataFrame(bench_data, columns=["x", "y", "z", "theta0", "theta1", "theta0_pred", "theta1_pred", "time_pred"])
    bench_csv.to_csv(f"runs/exp{args.exp_id}/esp32_bench.csv", index=False)

    print("Benchmark finished !!!")
    print("Benchmark results:")
    print("------------------")
    print(bench_csv.describe())