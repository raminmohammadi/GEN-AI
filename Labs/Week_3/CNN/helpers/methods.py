import platform
import tensorflow as tf
import os
import requests
import gzip
import numpy as np
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

# Loads the Fashion MNIST dataset
def load_data(data_dir: str):
    data_dir = data_dir
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    if not all(os.path.exists(os.path.join(data_dir, f)) for f in files):
        download_fashion_mnist(data_dir)

    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28, 1)

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    x_train = load_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    x_test = load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    return x_train, y_train, x_test, y_test

# Downloads the Fashion MNIST dataset
def download_fashion_mnist(data_dir):
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
    ]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for url in urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            r = requests.get(url, allow_redirects=True)
            open(filepath, 'wb').write(r.content)

    print('Fashion MNIST dataset downloaded successfully!')
    
# display image grid
def display_image_grid(images, labels):
  """
  Displays a grid of images with their corresponding labels.

  Args:
    images: A numpy array of images.
    labels: A numpy array of labels.
  """
  class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

  num_images = len(images)
  num_cols = 8
  num_rows = (num_images + num_cols - 1) // num_cols
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
  fig.subplots_adjust(hspace=0.5)
  for i in range(num_images):
    row = i // num_cols
    col = i % num_cols
    axes[row, col].imshow(images[i])
    # Access the first element of the array using labels[i][0]
    axes[row, col].set_title(f"Class: {class_names[labels[i][0]]}")
    axes[row, col].axis('off')
  plt.tight_layout()
  plt.show()
  
# Plotting the training history
def plot_training_history(history):
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

# Plotting the confusion matrix
def plot_predictions(predictions, x_test, y_test, class_names, num_rows=5, num_cols=3):
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(predictions, x_test, y_test, class_names, i)
    plt.tight_layout()
    plt.show()


# Plotting the image
def plot_image(predictions, x_test, y_test, class_names, i):
    true_label, img = y_test[i], x_test[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = int(true_label)

    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions[i]),
                                         class_names[true_label]),
               color=color)