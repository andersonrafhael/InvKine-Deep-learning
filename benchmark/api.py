import time
import requests
import argparse
import pandas as pd
from tqdm import tqdm


def get_api_prediction(url:str, data: dict) -> tuple[float, list[float]]:
        
    """
    This function sends to the API a message to get the prediction of the pose.
    
    Params:
    -------
    data: dict
        dictionary with the pose to be predicted
    
    """
    
    begin_time = time.time()
    response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
    end_time = time.time() - begin_time
    
    return end_time, response.json()
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Api inference Benchmark")
    parser.add_argument("--exp-id", type=int, default=0, help="Experiment id")
    args = parser.parse_args()
    
    URL = "https://invkine-model-fastapi-production.up.railway.app/inference"
    test_data = pd.read_csv(f"runs/exp{args.exp_id}/test.csv")
    
    api_bench_data = []
    for i, row in tqdm(test_data.iloc[:10,:].iterrows(), total=test_data.shape[0], desc="Running Api benchmark..."):
        pose_str = ",".join([str(value) for value in row.values[:3]])
        data = {
            "x": float(row.values[0]),
            "y": float(row.values[1]),
            "z": float(row.values[2])
        }
        
        api_time, api_thetas = get_api_prediction(URL, data)
        api_bench_data.append([row.x, row.y, row.z, row.theta0, row.theta1, api_thetas["theta0"], api_thetas["theta1"], api_time])
        
    api_bench_df = pd.DataFrame(api_bench_data, columns=["x", "y", "z", "theta0", "theta1", "theta0_pred", "theta1_pred", "time_pred"])
    api_bench_df.to_csv(f"runs/exp{args.exp_id}/api_bench.csv", index=False)