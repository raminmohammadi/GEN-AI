import numpy as np
import tensorflow as tf

# Test for pad_sequences_data
def test_pad_sequences_data(x_train_padded, x_test_padded):
    assert x_train_padded.shape == (25000, 200), "Incorrect shape for padded training data"
    assert x_test_padded.shape == (25000, 200), "Incorrect shape for padded test data"
    assert np.all(x_train_padded[:, -1] > 0), "Training padding failed"
    assert np.all(x_test_padded[:, -1] > 0), "Testing padding failed"
    print("test_pad_sequences_data passed")

# Test for create_model
def test_create_model(model):
    assert isinstance(model, tf.keras.Model), "Model should be an instance of tf.keras.Model"
    assert len(model.layers) > 0, "Model should have layers"
    assert isinstance(model.layers[0], tf.keras.layers.Embedding), "First layer should be an Embedding layer"
    embedding_layer = model.layers[0]
    
    assert embedding_layer.input_dim == 10000, "Incorrect input dimension for Embedding layer"
    assert embedding_layer.output_dim == 32, "Incorrect output dimension for Embedding layer"    
    print("test_create_model passed")

# Test for train_model
def test_train_model(history):
    assert 'accuracy' in history.history, "Training should track accuracy"
    assert len(history.history['loss']) > 0, "Training should record loss values"
    print("test_train_model passed")

# Test for evaluate_model
def test_evaluate_model(loss, accuracy):
    assert loss >= 0, "Loss should be non-negative"
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    print("test_evaluate_model passed")

