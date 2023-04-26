import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import argparse

def get_experiment_id(root_dir: str) -> int:
    """
    This function returns the path of the next experiment to be saved.

    Params:
    --------
    root_dir: str
        root directory of the project

    Returns:
    --------
    ith_experiment_id: int
        id of the current experiment

    """
    experiment_id_path = Path(root_dir) / ("run_id.json")

    if experiment_id_path.exists():
        current_id = json.load(open(experiment_id_path))["current_id"]
    else:
        current_id = 0

    with open(experiment_id_path, "w") as f:
        json.dump({"current_id": current_id + 1}, f, indent=4)

    return current_id

def get_experiment_config(args: argparse.Namespace) -> dict:
    
    experiment_config = {
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.bsize,
            "learning_rate": args.lrate,
            "n_samples": args.n_samples,
        },
    }
    
    return experiment_config

def get_network_config(input_dim:int, output_dim:int, neuron_layers: list[int]) -> list[int]:
    
    """
    This function returns the network configuration.
    
    Params:
    --------
    input_dim: int
        input dimension of the network
    output_dim: int
        output dimension of the network
    neuron_layers: list[int]
        list of neurons in each layer excluding the output layer
        
    Returns:
    --------
    net_config: list[int]
        network configuration including the input and output dimensions

    """
    
    net_config = [input_dim] + neuron_layers + [output_dim]
    return net_config