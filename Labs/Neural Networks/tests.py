import numpy as np
import tensorflow as tf

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

def test_model_accuracy(history, min_val_acc=0.44):
  """Tests if the model achieved at least the minimum validation accuracy.

  Args:
      history: The history object returned by the model's fit method.
      min_val_acc: The minimum required validation accuracy.

  Raises:
      AssertionError if the minimum validation accuracy is not met.
  """
  val_acc = max(history.history['val_accuracy'])
  assert val_acc >= min_val_acc, f"Model did not achieve minimum validation accuracy of {min_val_acc:.2f}, achieved {val_acc:.2f}"