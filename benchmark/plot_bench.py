import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from robot import Robot3DOF

def plot_scatter_xyz(xyz: list[list[float]], xyz_pred: list[list[float]], title: str):
    """
    This function plots the comparison between the true xyz and the predicted xyz

    Params:
    -------
    xyz: list[list[float]]
        list with the true xyz
    xyz_pred: list[list[float]]
        list with the predicted xyz
    title: str
        title of the plot
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Scatter plot " + title)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], label="xyz")
    ax.scatter(xyz_pred[:, 0], xyz_pred[:, 1], xyz_pred[:, 2], label="xyz_pred")
    ax.legend()
    plt.show()

    return

def plot_thetas(thetas: list[list[float]], thetas_pred: list[list[float]], title: str):
    """
    This function plots the comparison between the true thetas and the predicted thetas

    Params:
    -------
    thetas: list[list[float]]
        list with the true thetas
    thetas_pred: list[list[float]]
        list with the predicted thetas
    title: str
        title of the plot
    """

    for i in range(len(thetas)):
        plt.figure(figsize=(10, 8))
        plt.title(f"theta{i} vs theta{i}_pred " + title)
        plt.plot(thetas[i], label=f"theta{i} true")
        plt.plot(thetas_pred[i], label=f"theta{i}_pred")
        plt.legend()
        plt.show()
    return

def plot_time_prediction(inf_times: list[float], pred_times: list[float], title: str):
    """
    This function plots the comparison between the inference time and the prediction time which includes the communication time

    Params:
    -------
    inf_times: list[float]
        list with the inference times
    pred_times: list[float]
        list with the prediction times
    title: str
        title of the plot
    """

    m = [np.array(inf_times).mean(axis=0)] + [np.array(pred_times).mean(axis=0)]

    plt.figure(figsize=(10, 8))
    plt.title("Prediction time " + title)
    bp = plt.boxplot([inf_times, pred_times], labels=["inf_times", "pred_times"], showmeans=True)

    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = f' μ={m[i]}'
        plt.annotate(text, xy=(x, y))
    plt.legend()
    plt.show()
    return

def plot_time_comparison(api_pred_times: list[float], api_inf_times: list[float], esp_pred_times: list[float], esp_inf_times: list[float]):
    """
    This function plots the prediction time comparison between the API and the ESP32
    
    Params:
    -------
    api_pred_times: list[float]
        list with the api prediction times including the communication time
    api_inf_times: list[float]
        list with the api inference times
    esp_pred_times: list[float]
        list with the esp prediction times including the communication time
    esp_inf_times: list[float]
        list with the esp inference times
    """
    
    m = [
        np.array(api_pred_times).mean(axis=0), 
        np.array(api_inf_times).mean(axis=0), 
        np.array(esp_pred_times).mean(axis=0), 
        np.array(esp_inf_times).mean(axis=0)
    ]

    plt.figure(figsize=(10, 8))
    plt.title("Prediction time comparison")
    bp = plt.boxplot(
        [api_pred_times, api_inf_times, esp_pred_times, esp_inf_times], 
        labels=["API prediction time", "API inference time", "Esp prediction time", "Esp inference time"], 
        showmeans=True
    )

    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = f' μ={m[i]}'
        plt.annotate(text, xy=(x, y))
    plt.legend()
    plt.show()
    return

def plot_bench(test_data: pd.DataFrame, joints_number: int, title: str):

    """
    This function performs the benchmark plots
    
    Params:
    -------
    test_data: pd.DataFrame
        pandas dataframe with the benchmark data
    joints_number: int
        number of joints of the robot
    title: str
        title of the plots
    """

    robot = Robot3DOF()

    xyz = np.array([test_data['x'].values.tolist(), test_data['y'].values.tolist(), test_data['z'].values.tolist()]).reshape(-1, 3)
    
    thetas = [test_data[f"theta{i}"].values.tolist() for i in range(joints_number)]
    thetas_pred = np.array([test_data[f"theta{i}_pred"].values.tolist() for i in range(joints_number)])

    xyz_pred = np.array([robot.get_position(theta) for theta in thetas_pred.reshape(-1, 3)])

    plot_scatter_xyz(xyz, xyz_pred, title=title)
    plot_thetas(thetas, thetas_pred, title=title)
    plot_time_prediction(test_data["time_inf"].values.tolist(), test_data["time_pred"].values.tolist(), title=title)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Api inference Benchmark")
    parser.add_argument("--exp-id", type=str, default="3dof-64-64-64", help="Experiment id")
    # parser.add_argument("--exp-id", type=int, default=0, help="Experiment id")
    args = parser.parse_args()
    
    test_dataAPI = pd.read_csv(f"results/{args.exp_id}/api_bench.csv")
    test_dataESP = pd.read_csv(f"results/{args.exp_id}/esp32_bench.csv")

    plot_bench(test_dataAPI, joints_number=3, title="API")
    plot_bench(test_dataESP, joints_number=3, title="ESP32")
    plot_time_comparison(test_dataAPI["time_pred"].values.tolist(), test_dataAPI["time_inf"].values.tolist(), test_dataESP["time_pred"].values.tolist(), test_dataESP["time_inf"].values.tolist())