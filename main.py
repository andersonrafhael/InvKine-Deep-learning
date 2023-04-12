import numpy as np
from pprint import pprint

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from robot import RobotPuma560

np.set_printoptions(precision=6, suppress=True)

dh_table = np.array([
    
    [np.pi/2,   -np.pi/2,          0,      0.67183],
    [      0,           0,    0.43180,     0.13970],
    [      0,     np.pi/2,   -0.02032,           0],
    [      0,    -np.pi/2,         0,      0.43180],
    [      0,     np.pi/2,         0,            0],
    [      0,           0,         0,      0.05650]
    
])

robot = RobotPuma560(dh_table)