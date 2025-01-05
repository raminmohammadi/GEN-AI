import tensorflow as tf
from transformers import DistilBertTokenizerFast
from datasets import load_dataset
from helpers import *


def test_preprocessing():
    """Tests the preprocessing function."""
    try:
        sample_data = {
            "question": ["What is the capital of France?"],
            "context": ["Paris is the capital of France."],
            "answers": [{"text": ["Paris"], "answer_start": [0]}],
        }
        
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        processed = preprocess_function_(sample_data, tokenizer)
        required_keys = ['start_positions', 'end_positions']
        for key in required_keys:
            assert key in processed, f"Missing key: {key}"
        print("✓ Preprocessing test passed!")
    except Exception as e:
        raise ValueError(f"❌ Preprocessing test failed: {str(e)}")

def test_model_creation():
    """Tests model initialization."""
    try:
        model = create_qa_model_()
        assert hasattr(model, 'optimizer'), "Model not compiled with optimizer"
        
        # Ensure that the model has an optimizer set
        optimizer = model.optimizer
        
        batch_size = 2
        seq_length = 384
        dummy_input = {
            'input_ids': tf.zeros((batch_size, seq_length), dtype=tf.int32),
            'attention_mask': tf.ones((batch_size, seq_length), dtype=tf.int32),
        }
        _ = model(dummy_input, training=False)
        print("✓ Model creation test passed!")
    except Exception as e:
        raise ValueError(f"❌ Model creation test failed: {str(e)}")

def run_all_tests():
    """Runs all test functions."""
    print("Running all tests...\n")
    test_preprocessing()
    test_model_creation()
    print("\nAll tests completed.")

def test_model_loss(test_loss, min_val_loss=2.0):
  """Tests if the model achieved at least the minimum test loss.

  Args:
      test_accuracy: The accuracy of the model on the test data.
      min_val_loss: The minimum required validation loss.

  Raises:
      AssertionError if the minimum validation loss is not met.
  """
  assert test_loss <= min_val_loss, f"Model did not achieve minimum test loss of {min_val_loss:.2f}, achieved {test_loss:.2f}"
