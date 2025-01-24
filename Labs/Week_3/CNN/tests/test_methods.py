# Test functions
import numpy as np
import pytest

def test_preprocess_data(x_train_processed, y_train_processed, x_test_processed, y_test_processed):
    # Check normalization
    assert np.max(x_train_processed) <= 1.0 and np.min(x_train_processed) >= 0.0, "Training data should be normalized"
    assert np.max(x_test_processed) <= 1.0 and np.min(x_test_processed) >= 0.0, "Test data should be normalized"

def test_create_model(model):
    # Check if the model is created successfully
    assert model is not None, "Model should be created"
    assert len(model.layers) >= 4, "Model should have at least 4 layers"
    
    layer_types = [type(layer).__name__ for layer in model.layers]
    assert 'Conv2D' in layer_types, "Model should have at least one Conv2D layer"
    assert 'Dense' in layer_types, "Model should have at least one Dense layer"
    assert 'Flatten' in layer_types, "Model should have a Flatten layer"

def test_train_model(history):
    # Check if the history contains accuracy
    assert 'accuracy' in history.history, "Training accuracy should be recorded"
    assert len(history.history['accuracy']) >= 1, "History should have at least 1 epoch of training"

def test_evaluate_model(test_loss, test_acc):
    # Check if the evaluation returns correct values
    assert isinstance(test_loss, float), "Test loss should be a float"
    assert isinstance(test_acc, float), "Test accuracy should be a float"
    assert test_acc >= 0.0 and test_acc <= 1.0, "Test accuracy should be between 0 and 1"

def test_make_predictions(predictions):
    # Check if predictions are made and have the correct shape
    assert predictions is not None, "Predictions should not be None"
    assert predictions.shape == (10000, 10), "Predictions should have shape (10000, 10)"
    assert np.sum(predictions) >= 0, "Predictions should not be negative"