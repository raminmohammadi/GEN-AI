import tensorflow as tf
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd

# Detects and sets the device to be used for training
def detect_and_set_device():
    if tf.test.is_built_with_cuda():
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print("GPU is available. Using GPU.")
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return 'GPU'
            except RuntimeError as e:
                print(f"Unable to set memory growth: {e}")
                return 'CPU'
    print("GPU is not available. Using CPU.")
    return 'CPU'

# Data Loading 
def load_emotion_dataset():
    dataset = load_dataset("emotion")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    return train_df, test_df