import pandas as pd
import tensorflow as tf
import numpy as np
import pytest


def test_preprocess_data(X_train, X_test, y_train, y_test, tokenizer, emotion_labels, num_classes):
    # Assertions
    assert X_train.shape[1] == 50, "X_train should have max_seq_length of 10"
    assert X_test.shape[1] == 50, "X_test should have max_seq_length of 10"
    assert len(y_train) == X_train.shape[0], "y_train length should match X_train"
    assert len(y_test) == X_test.shape[0], "y_test length should match X_test"
    assert tokenizer.num_words <= 10000, "Vocabulary size should not exceed max_vocab_size"
    assert set(emotion_labels) == set([0, 1, 2, 3, 4, 5]), "Emotion labels should be [0, 1, 2, 3, 4, 5]"
    assert num_classes == 6, "Number of classes should be 6"

    # Check if oversampling worked
    unique, counts = np.unique(y_train, return_counts=True)
    assert len(set(counts)) == 6, "After oversampling, all classes should have the same count"

    print("preprocess_data test passed.")
    

def test_positional_encoding(pe):

    # Test get_angles method
    def test_get_angles():
        # Test case 1: Check angles for position 0
        angles_0 = pe.get_angles(0, 0, 512)
        expected_angle_0 = 0
        tf.debugging.assert_near(angles_0, expected_angle_0, message="Test failed for get_angles at position 0")

        # Test case 2: Check angles for position 1
        angles_1 = pe.get_angles(1, 0, 512)
        expected_angle_1 = 1 / (10000 ** (0 / 512))
        tf.debugging.assert_near(angles_1, expected_angle_1, message="Test failed for get_angles at position 1")

        # Test case 3: Check angles for position 2
        angles_2 = pe.get_angles(2, 2, 512)
        expected_angle_2 = 2 / (10000 ** (2 / 512))
        tf.debugging.assert_near(angles_2, expected_angle_2, message="Test failed for get_angles at position 2")

    # Test positional_encoding method
    def test_positional_encoding():
        # Test case 1: Check positional encoding for position 5 and dimension 4
        pos_enc = pe.positional_encoding(5, 4)
        expected_shape = (1, 5, 4)
        tf.debugging.assert_equal(tf.shape(pos_enc), expected_shape, message=f"Test failed: expected shape {expected_shape}, got {tf.shape(pos_enc)}")

        # Test case 2: Check values in positional encoding at position 0
        first_position = pos_enc[0, 0, :]
        expected_sine_values = tf.constant([0., 0.], dtype=tf.float32)  # sin(0) = 0 for both
        expected_cosine_values = tf.constant([1., 1.], dtype=tf.float32)  # cos(0) = 1 for both
        expected_values = tf.concat([expected_sine_values, expected_cosine_values], axis=-1)
        tf.debugging.assert_near(first_position, expected_values, message=f"Test failed: expected {expected_values}, got {first_position}")

        # Test case 3: Check for maximum position
        pos_enc_max = pe.positional_encoding(100, 512)
        tf.debugging.assert_equal(tf.shape(pos_enc_max), (1, 100, 512), message="Test failed: shape mismatch for maximum position")

    # Run the tests
    test_get_angles()
    test_positional_encoding()

    return "All positional encoding tests passed."
    
def test_multi_head_attention(mha):
    # Test parameters
    batch_size = 32
    seq_length = 50
    
    d_model = mha.d_model
    num_heads = mha.num_heads

    # Create sample input tensors
    query = tf.random.uniform((batch_size, seq_length, d_model))
    key = tf.random.uniform((batch_size, seq_length, d_model))
    value = tf.random.uniform((batch_size, seq_length, d_model))
    mask = None  # For simplicity, we're not using a mask in these tests

    # Test case 1: Check output shape
    output = mha(value, key, query, mask)
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

    # Test case 2: Check if the output is different from the input
    assert not np.allclose(output.numpy(), query.numpy()), "Output should be different from the input query"

    # Test case 3: Check if the layer is trainable
    assert len(mha.trainable_variables) > 0, "MultiHeadAttention layer should have trainable variables"

    # Test case 4: Check if splitting heads works correctly
    q_split = mha.split_heads(query, batch_size)
    expected_split_shape = (batch_size, num_heads, seq_length, d_model // num_heads)
    assert q_split.shape == expected_split_shape, f"Expected split shape {expected_split_shape}, but got {q_split.shape}"

    # Test case 5: Check if the output is consistent for the same input
    output1 = mha(value, key, query, mask)
    output2 = mha(value, key, query, mask)
    np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)

    # Test case 6: Check if the layer can handle different sequence lengths
    query_short = tf.random.uniform((batch_size, 5, d_model))
    output_short = mha(value, key, query_short, mask)
    expected_shape_short = (batch_size, 5, d_model)
    assert output_short.shape == expected_shape_short, f"Expected shape {expected_shape_short}, but got {output_short.shape}"

    return "All test cases passed of Multi-head attention block!"


def test_transformer_block(transformer_block):
    # Test parameters
    batch_size = 32
    seq_length = 50
    d_model = 128


    # Create sample input tensor
    x = tf.random.uniform((batch_size, seq_length, d_model))
    mask = None  # For simplicity, we're not using a mask in these tests

    # Test case 1: Check output shape
    output = transformer_block(x, training=False, mask=mask)
    expected_shape = (batch_size, seq_length, d_model)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Test case 1 (output shape) passed")

    # Test case 2: Check if the output is different from the input
    assert not np.allclose(output.numpy(), x.numpy()), "Output should be different from the input"
    print("Test case 2 (output difference) passed")

    # Test case 3: Check if the layer is trainable
    assert len(transformer_block.trainable_variables) > 0, "TransformerBlock should have trainable variables"
    print("Test case 3 (trainable variables) passed")

    # Test case 4: Check if dropout is applied during training
    output_train = transformer_block(x, training=True, mask=mask)
    output_test = transformer_block(x, training=False, mask=mask)
    assert not np.allclose(output_train.numpy(), output_test.numpy()), "Dropout should cause different outputs in training and testing modes"
    print("Test case 4 (dropout effect) passed")

    # Test case 5: Check if the output is consistent for the same input in test mode
    output1 = transformer_block(x, training=False, mask=mask)
    output2 = transformer_block(x, training=False, mask=mask)
    np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)
    print("Test case 5 (consistency in test mode) passed")
    

    # Test case 6: Check if the block can handle different sequence lengths
    x_short = tf.random.uniform((batch_size, 5, d_model))
    output_short = transformer_block(x_short, training=False, mask=mask)
    expected_shape_short = (batch_size, 5, d_model)
    assert output_short.shape == expected_shape_short, f"Expected shape {expected_shape_short}, but got {output_short.shape}"
    print("Test case 8 (variable sequence length) passed")

    print("All transformer block test cases passed!")


def test_transformer_model(model):
    # Test parameters
    num_layers = model.num_layers
    input_vocab_size = model.input_vocab_size
    num_classes = model.num_classes
    batch_size = 32
    seq_length = 50

    # Create sample input tensor
    x = tf.random.uniform((batch_size, seq_length), dtype=tf.int32, maxval=input_vocab_size)

    # Test case 1: Check output shape
    output = model(x, training=False)
    expected_shape = (batch_size, num_classes)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    print("Test case 1 (output shape) passed")

    # Test case 2: Check if the output is a valid probability distribution
    assert np.allclose(np.sum(output.numpy(), axis=1), 1.0), "Output should be a valid probability distribution"
    assert np.all(output.numpy() >= 0) and np.all(output.numpy() <= 1), "Output values should be between 0 and 1"
    print("Test case 2 (valid probability distribution) passed")

    # Test case 3: Check if the model is trainable
    assert len(model.trainable_variables) > 0, "TransformerModel should have trainable variables"
    print("Test case 3 (trainable variables) passed")

    # Test case 4: Check if dropout is applied during training
    output_train = model(x, training=True)
    output_test = model(x, training=False)
    assert not np.allclose(output_train.numpy(), output_test.numpy()), "Dropout should cause different outputs in training and testing modes"
    print("Test case 4 (dropout effect) passed")

    # Test case 5: Check if the output is consistent for the same input in test mode
    output1 = model(x, training=False)
    output2 = model(x, training=False)
    np.testing.assert_allclose(output1.numpy(), output2.numpy(), atol=1e-5)
    print("Test case 5 (consistency in test mode) passed")

    # Test case 6: Check if embedding layer is applied
    embedding_vars = [var for var in model.trainable_variables if 'embedding' in var.name]
    assert len(embedding_vars) > 0, "Embedding layer variables should exist"
    print("Test case 6 (embedding layer) passed")

    # Test case 7: Check if positional encoding is applied
    assert hasattr(model, 'pos_encoding'), "Positional encoding should be a part of the model"
    print("Test case 7 (positional encoding) passed")

    # Test case 8: Check if the model can handle different sequence lengths
    x_short = tf.random.uniform((batch_size, 30), dtype=tf.int32, maxval=input_vocab_size)
    output_short = model(x_short, training=False)
    assert output_short.shape == expected_shape, f"Expected shape {expected_shape}, but got {output_short.shape}"
    print("Test case 8 (variable sequence length) passed")

    # Test case 9: Check if the model has the correct number of transformer blocks
    assert len(model.transformer_blocks) == num_layers, f"Expected {num_layers} transformer blocks, but got {len(model.transformer_blocks)}"
    print("Test case 9 (number of transformer blocks) passed")

    # Test case 10: Check if the final layer has the correct number of units
    assert model.final_layer.units == num_classes, f"Expected {num_classes} units in the final layer, but got {model.final_layer.units}"
    print("Test case 10 (final layer units) passed")

    print("All transformer model test cases passed!")

def test_train_model(history):
    # Check if the history contains accuracy
    assert 'accuracy' in history.history, "Training accuracy should be recorded"
    assert len(history.history['accuracy']) >= 1, "History should have at least 1 epoch of training"
    
def test_evaluate_model(test_loss, test_acc):
    # Check if the evaluation returns correct values
    assert isinstance(test_loss, float), "Test loss should be a float"
    assert isinstance(test_acc, float), "Test accuracy should be a float"
    assert test_acc >= 0.0 and test_acc <= 1.0, "Test accuracy should be between 0 and 1"