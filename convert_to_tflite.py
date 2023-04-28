import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

if __name__ == "__main__":
    
    model = load_model("output/model.h5")
    
    print(model.predict([[np.pi, np.pi, 0]]))
    
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional optimization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    
    with open("output/model.tflite", "wb") as f:
        f.write(tflite_model)
    