import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_thetas_comparison(thetas: list[list[float]], thetas_pred: list[list[float]]):
    for i in range(len(thetas)):
        plt.figure(figsize=(10, 8))
        plt.title(f"theta{i} vs theta{i}_pred")
        plt.plot(thetas[i], label=f"theta{i} true")
        plt.plot(thetas_pred[i], label=f"theta{i}_pred")
        plt.legend()
        plt.show()
    return

def plot_thetas_error(thetas: list[list[float]], thetas_pred: list[list[float]]):
    for i in range(len(thetas)):
        plt.figure(figsize=(10, 8))
        plt.title(f"Erro theta{i} vs theta{i}_pred")
        erro = abs(np.array(thetas[i]) - np.array(thetas_pred[i]))
        plt.plot(erro, label=f"theta{i} erro")
        plt.legend()
        plt.show()
    return

def plot_time_prediction(inf_times: list[float], pred_times: list[float]):

    m = [np.array(inf_times).mean(axis=0)] + [np.array(pred_times).mean(axis=0)]

    plt.figure(figsize=(10, 8))
    plt.title("Prediction time")
    bp = plt.boxplot([inf_times, pred_times], labels=["inf_times", "pred_times"], showmeans=True)

    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = f' μ={m[i]}'
        plt.annotate(text, xy=(x, y))
    plt.legend()
    plt.show()
    return

def plot_time_comparison(apiTimes: list[float], espTimes: list[float]):
    m = [np.array(espTimes).mean(axis=0)] + [np.array(apiTimes).mean(axis=0)]

    plt.figure(figsize=(10, 8))
    plt.title("Prediction time")
    bp = plt.boxplot([espTimes, apiTimes], labels=["Esp time", "API time"], showmeans=True)

    for i, line in enumerate(bp['medians']):
        x, y = line.get_xydata()[1]
        text = f' μ={m[i]}'
        plt.annotate(text, xy=(x, y))
    plt.legend()
    plt.show()
    return

def plot_bench(test_data: pd.DataFrame):
    thetas = [test_data["theta0"].values.tolist(), test_data["theta1"].values.tolist(), test_data["theta2"].values.tolist()]
    thetas_pred = [test_data["theta0_pred"].values.tolist(), test_data["theta1_pred"].values.tolist(), test_data["theta2_pred"].values.tolist()]
    # plot_thetas_comparison(thetas, thetas_pred)
    plot_time_prediction(test_data["time_inf"].values.tolist(), test_data["time_pred"].values.tolist())
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Api inference Benchmark")
    parser.add_argument("--exp-id", type=str, default="3dof-64-64-64", help="Experiment id")
    # parser.add_argument("--exp-id", type=int, default=0, help="Experiment id")
    args = parser.parse_args()
    
    test_dataAPI = pd.read_csv(f"results/{args.exp_id}/api_bench.csv")
    test_dataESP = pd.read_csv(f"results/{args.exp_id}/esp32_bench.csv")

    plot_bench(test_dataAPI)
    # plot_bench(test_dataESP)
    plot_time_comparison(test_dataAPI["time_pred"].values.tolist(), test_dataESP["time_pred"].values.tolist())