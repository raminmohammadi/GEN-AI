import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test_vectorize_text(vectorize_layer):
    assert isinstance(vectorize_layer, tf.keras.layers.TextVectorization), "Should return a TextVectorization layer"
    
    # Check output_sequence_length from the layer's config
    assert vectorize_layer.get_config()['output_sequence_length'] == 100, "Output sequence length should be 100"
    
    # Check max_tokens from the layer's config
    assert vectorize_layer.get_config()['max_tokens'] == 20000, "Max tokens should be 20000"
    
    # Test the layer on some data
    test_input = tf.constant(['This is a new test'])
    test_output = vectorize_layer(test_input)
    assert test_output.shape == (1, 100), "Output shape should be (1, 100)"
    print("vectorize_text test passed")

def test_self_attention(attention_layer):
    input_tensor = tf.random.normal((32, 100, 128))
    output = attention_layer(input_tensor)
    assert output.shape == (32, 128), f"Expected shape (32, 128), but got {output.shape}"
    assert tf.reduce_all(tf.math.is_finite(output)), "Output contains NaN or infinity"
    print("SelfAttention test passed")

def test_create_model(model):
    assert isinstance(model, tf.keras.Model), "Should return a Keras Model"
    assert len(model.layers) == 7, f"Expected 7 layers, but got {len(model.layers)}"
    assert isinstance(model.layers[-1], tf.keras.layers.Dense), "Last layer should be Dense"
    assert model.layers[-1].units == 4, "Last layer should have 4 units"
    assert model.layers[-1].activation == tf.keras.activations.softmax, "Last layer should use softmax activation"
    print("create_model test passed")

def test_train_model(history):
    assert isinstance(history, tf.keras.callbacks.History), "Should return a History object"
    assert 'accuracy' in history.history, "History should contain accuracy"
    assert 'val_accuracy' in history.history, "History should contain validation accuracy"
    assert len(history.history['accuracy']) > 0, "Should have trained for at least one epoch"
    print("train_model test passed")

def test_evaluate_model(predicted_classes):
    assert isinstance(predicted_classes, np.ndarray), "Should return a numpy array"
    assert len(predicted_classes) == 7600, f"Expected 100 predictions, but got {len(predicted_classes)}"
    assert np.all((predicted_classes >= 0) & (predicted_classes < 4)), "Predicted classes should be between 0 and 3"
    print("evaluate_model test passed")