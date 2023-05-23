import json
import numpy as np
from pathlib import Path
import subprocess as sp
import argparse
from tinymlgen import port
import tensorflow as tf
from everywhereml.code_generators.tensorflow import tf_porter

def get_experiment_id(root_dir: str) -> int:
    """
    This function returns the id where the experiment must be saved.

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


def get_network_config(
    input_dim: int, output_dim: int, neuron_layers: list[int]
) -> list[int]:
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


def export_int8_model(
    model: tf.keras.models.Sequential,
    representative_pose: list[np.ndarray],
    experiment_folder: Path,
) -> None:
    """
    This function applies the int8 quantization to the model.

    Params:
    -------
    model: tf.keras.sequential
        model to be quantized
    representative_pose: list[np.ndarray]
        list of representative poses to be used for quantization
    """
    model.save(experiment_folder / "model.h5")

    representative_pose = representative_pose.astype(np.float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    n = representative_pose.shape[1]
    def representative_dataset_gen():
        for pose in representative_pose:
            yield [pose.reshape(1, n)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_quant_model = converter.convert()

    open(experiment_folder / "model.tflite", "wb").write(tflite_quant_model)

    sp.run(["xxd", "-i", f"{experiment_folder}/model.tflite", f"{experiment_folder}/model.cc"])
    
    
def export_H_model(model: tf.keras.models.Sequential, experiment_folder: Path):
    
    """
    This function exports the model to a C header file.
    
    Parameters:
    -----------
    model: tf.keras.sequential
        model to be exported
    experiment_folder: Path
        path to save the model
    """
    model_h_format = port(model, pretty_print=True, variable_name='invkine_model')
    with open(experiment_folder / "invkine_model.h", "w") as f:
        f.write(model_h_format)
        