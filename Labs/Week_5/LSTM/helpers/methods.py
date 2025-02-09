import platform
import tensorflow as tf
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# Detects and sets the device to be used for training
def detect_and_set_device():
    if tf.test.is_built_with_cuda() or platform.system() == "Darwin":
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

def load_imdb_data(num_words):
    """Load the IMDB dataset."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    print('Data Loaded Successfully')
    return (x_train, y_train), (x_test, y_test)

# Plotting the training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()