import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Setup
def detect_and_set_device():
    """Detects if a GPU is available and sets the device accordingly.

    Returns:
        str: 'GPU' if a GPU is available and memory growth is set successfully,
            otherwise 'CPU'.
    """
    return 'GPU' if tf.test.is_gpu_available() else 'CPU'

# Tests
def preprocessing_tests(X_train, X_test, y_train, y_test, num_classes=10):
  """Performs preprocessing tests on the given data.

  Args:
    X_train: Training data.
    X_test: Testing data.
    y_train: Training labels.
    y_test: Testing labels.
    num_classes: Number of classes.

  Raises:
    AssertionError if any of the tests fail.
  """

  # Check if samples are between 0 and 1
  assert np.all(0 <= X_train) and np.all(X_train <= 1), "Training samples should be between 0 and 1."
  assert np.all(0 <= X_test) and np.all(X_test <= 1), "Testing samples should be between 0 and 1."

  # Check if labels are one-hot encoded
  assert y_train.shape[-1] == num_classes, f"Training labels should have {num_classes} columns."
  assert y_test.shape[-1] == num_classes, f"Testing labels should have {num_classes} columns."
  assert np.all(np.isclose(np.sum(y_train, axis=1), 1)), "Training labels should sum to 1."
  assert np.all(np.isclose(np.sum(y_test, axis=1), 1)), "Testing labels should sum to 1."  

def test_model_structure(model):
    """Tests if the model has at least 3 hidden layers and the final layer has the specified number of neurons.

    Args:
        model: The Keras model to test.
        num_classes: The expected number of neurons in the final layer.
    """
    hidden_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    assert len(hidden_layers) >= 3, "Model should have at least 3 hidden layers."

    final_layer = hidden_layers[-1]  # Get the last hidden layer (which is the output layer)
    assert final_layer.units == 10, f"Final layer should have 10 neurons."

def test_model_compilation(model):
  """Tests if the model is compiled.

  Args:
      model(tf.keras.Model): The model to be tested.
  """
  assert model.compiled, "Model is not compiled."

# Plots
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

def plot_metrics(history):
  # Plotting training & validation accuracy values
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Train Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc='upper left')

  # Plotting training & validation loss values
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Train Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc='upper left')

  plt.tight_layout()
  plt.show()

def plot_label_comparison(y_true, y_pred):
  """Plots a bar chart comparing the total number of predicted labels that were equal to the total number of true labels.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.
  """
  class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

  # Calculate the confusion matrix
  cm = confusion_matrix(y_true, y_pred)

  # Get the diagonal elements of the confusion matrix (correct predictions)
  correct_predictions = cm.diagonal()

  # Sort the correct predictions and corresponding labels
  sorted_indices = np.argsort(correct_predictions)
  sorted_predictions = correct_predictions[sorted_indices]
  sorted_labels = np.arange(len(correct_predictions))[sorted_indices]

  # Create a bar plot
  plt.figure(figsize=(10, 6))
  plt.bar(sorted_labels, sorted_predictions)
  plt.xlabel("Class Label")
  plt.ylabel("Count")
  plt.title("Comparison of Predicted and True Labels")
  plt.xticks(sorted_labels, [class_names[i] for i in sorted_labels])
  plt.show()

import matplotlib.pyplot as plt

def plot_predictions(x_test, y_true, y_pred, num_samples=10):
  """Plots a grid of images with true and predicted labels, color-coded based on accuracy.

  Args:
    x_test: Test images.
    y_true: True labels.
    y_pred: Predicted labels.
    num_samples: Number of samples to plot.
  """

  class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

  num_plots = min(num_samples, len(y_true))
  fig, axes = plt.subplots(nrows=int(np.ceil(np.sqrt(num_plots))), ncols=int(np.ceil(np.sqrt(num_plots))), figsize=(15, 15))

  for i in range(num_plots):
    ax = axes.flatten()[i]
    ax.imshow(x_test[i], cmap='gray')

    # Determine the color based on accuracy
    if y_true[i] == y_pred[i]:
      color = 'green'
    else:
      color = 'red'

    # Set the labels with the appropriate color and adjust padding
    ax.set_xlabel(f"True: {class_names[y_true[i]]} \n Predicted: {class_names[y_pred[i]]}", color=color, fontsize=12, ha='center', va='top')

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing as needed

  plt.show()