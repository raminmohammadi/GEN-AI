import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class VAEValidator:
    @staticmethod
    def validate_dataset(dataset) -> bool:
        """Validates if the dataset is properly configured"""
        if not isinstance(dataset, tf.data.Dataset):
            raise ValueError("Dataset must be a tf.data.Dataset instance")

        # Check if dataset elements have correct shape and type
        for batch in dataset.take(1):
            if len(batch.shape) != 4:  # [batch_size, height, width, channels]
                raise ValueError(f"Expected 4D tensor, got shape {batch.shape}")
            if batch.dtype != tf.float32:
                raise ValueError(f"Expected float32 dtype, got {batch.dtype}")
            # if not (0 <= tf.reduce_min(batch) <= tf.reduce_max(batch) <= 1.0):
            #     raise ValueError("Image values should be normalized between 0 and 1")
        return True

    @staticmethod
    def validate_encoder(encoder, input_shape: Tuple[int, int, int], latent_dim: int) -> bool:
        """Validates encoder architecture"""
        try:
            # Test with dummy input
            test_input = tf.random.normal([1] + list(input_shape))
            z_mean, z_log_var = encoder(test_input)

            if not (isinstance(z_mean, tf.Tensor) and isinstance(z_log_var, tf.Tensor)):
                raise ValueError("Encoder must return two tensors (z_mean, z_log_var)")

            if z_mean.shape[1] != latent_dim or z_log_var.shape[1] != latent_dim:
                raise ValueError(f"Encoder output dimension must match latent_dim ({latent_dim})")

            return True
        except Exception as e:
            raise ValueError(f"Encoder validation failed: {str(e)}")

    @staticmethod
    def validate_sampling(sampling_layer, latent_dim: int) -> bool:
        """Validates sampling layer implementation"""
        try:
            # Test with dummy inputs
            test_mean = tf.random.normal([1, latent_dim])
            test_log_var = tf.random.normal([1, latent_dim])

            result = sampling_layer([test_mean, test_log_var])

            if result.shape != test_mean.shape:
                raise ValueError(f"Sampling layer output shape {result.shape} doesn't match input shape {test_mean.shape}")

            return True
        except Exception as e:
            raise ValueError(f"Sampling layer validation failed: {str(e)}")

    @staticmethod
    def validate_decoder(decoder, latent_dim: int, output_shape: Tuple[int, int, int]) -> bool:
        """Validates decoder architecture"""
        try:
            # Test with dummy input
            test_input = tf.random.normal([1, latent_dim])
            output = decoder(test_input)

            expected_shape = (1,) + output_shape
            if output.shape != expected_shape:
                raise ValueError(f"Decoder output shape {output.shape} doesn't match expected shape {expected_shape}")

            return True
        except Exception as e:
            raise ValueError(f"Decoder validation failed: {str(e)}")

    @staticmethod
    def validate_vae_loss(loss_function, batch_size: int, input_shape: Tuple[int, int, int], latent_dim: int) -> bool:
        """Validates VAE loss function"""
        try:
            # Create dummy data
            inputs = tf.random.normal([batch_size] + list(input_shape))
            outputs = tf.random.normal([batch_size] + list(input_shape))
            z_mean = tf.random.normal([batch_size, latent_dim])
            z_log_var = tf.random.normal([batch_size, latent_dim])

            loss = loss_function(inputs, outputs, z_mean, z_log_var)

            if not isinstance(loss, tf.Tensor):
                raise ValueError("Loss function must return a tensor")
            if len(loss.shape) != 0:  # Should be a scalar
                raise ValueError("Loss function must return a scalar value")

            return True
        except Exception as e:
            raise ValueError(f"Loss function validation failed: {str(e)}")


def test_loss_value(loss_value):
  """Checks if the loss value is less than or equal to 100.

  Args:
    loss_value: The loss value to check.

  Raises:
    ValueError: If the loss value is greater than 100.
  """
  if loss_value > 100:
    raise ValueError("Test Failed ❌. Loss is greater than 100.")
  else:
    print("You successfully passed the test ✅. Loss value is less than or equal to 100.")