import numpy
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def validate_data_preparation(x_train_seqs, x_test_seqs, y_train, y_test):
    """Checks if data preparation is carried out as intended.

    Args:
        x_train_seqs: The tokenized and padded training data.
        x_test_seqs: The tokenized and padded test data.
        y_train: The one-hot encoded training labels.
        y_test: The one-hot encoded test labels.

    Raises:
        ValueError: If any of the checks fail.
    """

    # Check if the training data is tokenized and padded.
    if not isinstance(x_train_seqs, numpy.ndarray):
        raise ValueError("Training data is not a NumPy array.")
    if x_train_seqs.ndim != 2:
        raise ValueError("Training data is not 2-dimensional.")
    if x_train_seqs.dtype != numpy.int32:
        raise ValueError("Training data is not of type int32.")

    # Check if the test data is tokenized and padded.
    if not isinstance(x_test_seqs, numpy.ndarray):
        raise ValueError("Test data is not a NumPy array.")
    if x_test_seqs.ndim != 2:
        raise ValueError("Test data is not 2-dimensional.")
    if x_test_seqs.dtype != numpy.int32:
        raise ValueError("Test data is not of type int32.")

    # Check if the training labels are one-hot encoded.
    if not isinstance(y_train, numpy.ndarray):
        raise ValueError("Training labels are not a NumPy array.")
    if y_train.ndim != 2:
        raise ValueError("Training labels are not 2-dimensional.")
    # if y_train.dtype != numpy.int32:
    #     raise ValueError("Training labels are not of type int32.")

    # Check if the test labels are one-hot encoded.
    if not isinstance(y_test, numpy.ndarray):
        raise ValueError("Test labels are not a NumPy array.")
    if y_test.ndim != 2:
        raise ValueError("Test labels are not 2-dimensional.")
    # if y_test.dtype != numpy.int32:
    #     raise ValueError("Test labels are not of type int32.")

    # Check if the number of samples in the training and test data is consistent with the number of labels.
    if x_train_seqs.shape[0] != y_train.shape[0]:
        raise ValueError("Number of training samples does not match number of training labels.")
    if x_test_seqs.shape[0] != y_test.shape[0]:
        raise ValueError("Number of test samples does not match number of test labels.")

    # Check if the number of classes in the training and test labels is consistent.
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError("Number of classes in training labels does not match number of classes in test labels.")


def validate_multihead_attention_model(model, vocab_size, embedding_dim, num_heads, ff_dim):
    """Checks if the multihead_attention_model is built as intended.

    Args:
        model: The multihead_attention_model to check.
        vocab_size: The size of the vocabulary.
        embedding_dim: The dimension of the embedding layer.
        num_heads: The number of attention heads.
        ff_dim: The dimension of the feedforward layer.

    Raises:
        ValueError: If any of the checks fail.
    """

    # Check if the model is an instance of the Model class.
    if not isinstance(model, Model):
        raise ValueError("Model is not an instance of the Model class.")

    # Check if the model has the correct input shape.
    if model.input_shape != (None, None):
        raise ValueError("Model input shape is incorrect.")

    # Check if the model has the correct output shape.
    if model.output_shape != (None, 3):
        raise ValueError("Model output shape is incorrect.")

    # Check if the layers have the correct parameters.
    embedding_layer = model.layers[1]
    if embedding_layer.input_dim != vocab_size:
        raise ValueError("Embedding layer input dimension is incorrect.")
    if embedding_layer.output_dim != embedding_dim:
        raise ValueError("Embedding layer output dimension is incorrect.")

    attention_layer = model.layers[2]
    if attention_layer.num_heads != num_heads:
        raise ValueError("Multi-head attention layer number of heads is incorrect.")
    if attention_layer.key_dim != embedding_dim:
        raise ValueError("Multi-head attention layer key dimension is incorrect.")

    ff_layer1 = model.layers[4]
    if ff_layer1.units != ff_dim:
        raise ValueError("First feedforward layer units is incorrect.")

    ff_layer2 = model.layers[5]
    if ff_layer2.units != embedding_dim:
        raise ValueError("Second feedforward layer units is incorrect.")

    output_layer = model.layers[-1]
    if output_layer.units != 3:
        raise ValueError("Output layer units is incorrect.")
    if output_layer.activation.__name__ != "softmax":
        raise ValueError("Output layer activation is incorrect.")
    
def test_model_accuracy(test_accuracy, min_val_acc=0.8):
  """Tests if the model achieved at least the minimum test accuracy.

  Args:
      test_accuracy: The accuracy of the model on the test data.
      min_val_acc: The minimum required validation accuracy.

  Raises:
      AssertionError if the minimum validation accuracy is not met.
  """
  assert test_accuracy >= min_val_acc, f"Model did not achieve minimum test accuracy of {min_val_acc:.2f}, achieved {test_accuracy:.2f}"