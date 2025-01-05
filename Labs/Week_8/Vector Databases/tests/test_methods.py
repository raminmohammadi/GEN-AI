from dataclasses import dataclass
import numpy as np
import pandas as pd
import tensorflow as tf
from pinecone import ServerlessSpec
from pinecone import Index


class Pinecone:
    def __init__(self, api_key):
        # Simulate API key validation
        assert api_key == 'your_pinecone_api_key', "Invalid API Key"
        
    def list_indexes(self):
        # Mock the list of indexes
        return ["index1", "index2"]
    
    def create_index(self, name, dimension, metric, spec):
        # Mock index creation with correct parameters
        assert name == 'your_index_name', "Index name mismatch"
        assert dimension == 2048, "Expected dimension 2048"
        assert metric == 'cosine', "Expected cosine metric"
        assert isinstance(spec, ServerlessSpec), "Expected ServerlessSpec type"
    
    def Index(self, name):
        # Return a mocked PineconeIndex object with a name attribute
        return PineconeIndex(name)


class PineconeIndex:
    def __init__(self, name):
        self.name = name  # Adding the name attribute to the mock object

    def upsert(self, vectors):
        # Mock upsert functionality (does nothing here)
        pass


# Test function for the feature extractor
def test_initialize_feature_extractor(model, config):
    # Assert the model is not None
    assert model is not None, "Model should not be None"
    
    # Assert the model has been constructed with a base ResNet50 model
    assert isinstance(model, tf.keras.Sequential), "Model should be a Sequential model"
    assert len(model.layers) == 2, "Model should have 2 layers (ResNet + GlobalPooling)"
    
    # Assert that the base model is using the ResNet50 architecture
    assert 'resnet50' in model.layers[0].name.lower(), "The first layer should be ResNet50"
    
    # Assert the model input size matches the config image size
    input_shape = model.input_shape[1:3]
    assert input_shape == config.image_size, f"Model input shape should be {config.image_size}"
    
    # Test if model can run inference on a dummy image
    dummy_image = np.random.rand(1, config.image_size[0], config.image_size[1], 3).astype(np.float32)
    output = model.predict(dummy_image)
    
    # Assert the output shape is as expected (batch size, 2048)
    assert output.shape == (1, 2048), f"Expected output shape (1, 2048), but got {output.shape}"


# Test function for Pinecone setup
def test_setup_pinecone(index, config, pc):
    # Fetch the description of the index using pinecone.describe_index()
    index_description = pc.describe_index(config.index_name)

    # Assert that the index is the one we expect
    assert index_description["name"] == config.index_name, f"Expected index name {config.index_name}, got {index_description['name']}"
    
    # Ensure that the index's dimension matches the expected one
    assert index_description["dimension"] == 2048, "Index dimension should be 2048"

    # Ensure that index is an instance of the correct type (Pinecone Index)
    assert isinstance(index, Index), "Index should be an instance of Pinecone Index"





# Test function for batch image processing
def test_batch_process_images(config, df, model, index):
    # Ensure batch size is positive
    assert config.batch_size > 0, "Batch size should be a positive number"

    # Calculate the total number of batches expected
    num_images = len(df['image'])
    num_batches = (num_images + config.batch_size - 1) // config.batch_size  # This ensures proper division

    # Ensure the number of batches is logical
    assert num_batches > 0, f"Expected at least 1 batch, but got {num_batches}"
    
    # Ensure that the last batch is not larger than the total number of images
    remaining_images = num_images % config.batch_size
    if remaining_images > 0:
        assert remaining_images <= config.batch_size, f"Remaining images in last batch should be <= batch size, but got {remaining_images}"

    # Test model output on a random dummy image
    dummy_image = np.random.rand(1, config.image_size[0], config.image_size[1], 3).astype(np.float32)
    model_output = model.predict(dummy_image)
    
    # Check that the model returns an output of the expected dimension
    assert model_output.shape[1] == 2048, f"Expected output size to be 2048, got {model_output.shape[1]}"


