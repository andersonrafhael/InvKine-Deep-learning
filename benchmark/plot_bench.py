import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from robot import Robot3DOF

def plot_scatter_xyz(xyz: list[list[float]], xyz_pred: list[list[float]], title: str):
    print(xyz_pred)
    # código para fazer scatter plot 3d
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Scatter plot " + title)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], label="xyz")
    ax.scatter(xyz_pred[:, 0], xyz_pred[:, 1], xyz_pred[:, 2], label="xyz_pred")
    ax.legend()
    plt.show()

    return

def plot_thetas_comparison(thetas: list[list[float]], thetas_pred: list[list[float]], title: str):
    for i in range(len(thetas)):
        plt.figure(figsize=(10, 8))
        plt.title(f"theta{i} vs theta{i}_pred " + title)
        plt.plot(thetas[i], label=f"theta{i} true")
        plt.plot(thetas_pred[i], label=f"theta{i}_pred")
        plt.legend()
        plt.show()
    return

def plot_thetas_error(thetas: list[list[float]], thetas_pred: list[list[float]], title: str):
    for i in range(len(thetas)):
        plt.figure(figsize=(10, 8))
        plt.title(f"Erro theta{i} vs theta{i}_pred " + title)
        erro = abs(np.array(thetas[i]) - np.array(thetas_pred[i]))
        plt.plot(erro, label=f"theta{i} erro")
        plt.legend()
        plt.show()
    return

def plot_time_prediction(inf_times: list[float], pred_times: list[float], title: str):

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
    m = [
        np.array(api_pred_times).mean(axis=0) + 
        np.array(api_inf_times).mean(axis=0) + 
        np.array(esp_pred_times).mean(axis=0) + 
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

def plot_bench(test_data: pd.DataFrame, title: str):
    robot = Robot3DOF()
    thetas = [test_data["theta0"].values.tolist(), test_data["theta1"].values.tolist(), test_data["theta2"].values.tolist()]
    # thetas_pred = [test_data["theta0_pred"].values.tolist(), test_data["theta1_pred"].values.tolist(), test_data["theta2_pred"].values.tolist()]
    # plot_thetas_comparison(thetas, thetas_pred, title=title)
    xyz = test_data.values[0:3].reshape(-1, 3)
    
    print(xyz.shape)
    thetas_pred = test_data.values[6:9].reshape(-1, 3)

    xyz_pred = np.array([robot.get_position(theta) for theta in thetas_pred])
    print(xyz_pred.shape)
    plot_scatter_xyz(xyz, xyz_pred, title=title)
    plot_time_prediction(test_data["time_inf"].values.tolist(), test_data["time_pred"].values.tolist(), title=title)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Api inference Benchmark")
    parser.add_argument("--exp-id", type=str, default="3dof-64-64-64", help="Experiment id")
    # parser.add_argument("--exp-id", type=int, default=0, help="Experiment id")
    args = parser.parse_args()
    
    test_dataAPI = pd.read_csv(f"results/{args.exp_id}/api_bench.csv")
    test_dataESP = pd.read_csv(f"results/{args.exp_id}/esp32_bench.csv")

    plot_bench(test_dataAPI, title="API")
    # plot_bench(test_dataESP, title="ESP32")
    plot_time_comparison(test_dataAPI["time_pred"].values.tolist(), test_dataAPI["time_pred"].values.tolist(), test_dataESP["time_pred"].values.tolist())