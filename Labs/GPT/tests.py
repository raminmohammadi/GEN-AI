
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from datasets import load_dataset
import numpy as np

def validate_shakespeare_generator(generator):
    """
    Validates the ShakespeareGenerator by checking each component
    Raises appropriate exceptions if any validation fails
    """
    def check_data_preparation():
        if not hasattr(generator, 'train_texts'):
            raise AttributeError("Dataset not loaded properly")
        
        if not hasattr(generator, 'tokenizer'):
            raise AttributeError("Tokenizer not initialized")
            
        if not generator.tokenizer.pad_token:
            raise ValueError("Tokenizer pad token not set")
            
        if not hasattr(generator, 'train_encodings'):
            raise AttributeError("Text encodings not created")
            
        # Check encodings shape and content
        if 'input_ids' not in generator.train_encodings:
            raise KeyError("input_ids missing from encodings")
            
        if 'attention_mask' not in generator.train_encodings:
            raise KeyError("attention_mask missing from encodings")
            
        # Validate dataset
        if not isinstance(generator.train_dataset, tf.data.Dataset):
            raise TypeError("train_dataset is not a TensorFlow Dataset")
        
        # Check dataset shapes
        for batch in generator.train_dataset.take(1):
            if len(batch) != 3:  # input_ids, attention_mask, labels
                raise ValueError("Dataset batch structure is incorrect")
            if batch[0].shape[1] != generator.max_length:
                raise ValueError(f"Sequence length mismatch. Expected {generator.max_length}")
    try:
        print("\n\nValidating data preparation...")
        check_data_preparation()
        print("✓ Data preparation validated successfully")

        return True

    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        raise

def check_model_setup(generator, model):
    print("\n\nValidating model setup...")

    if not isinstance(model, TFGPT2LMHeadModel):
        raise TypeError("Model is not an instance of TFGPT2LMHeadModel")
        
    # Check if model is compiled
    if not model.optimizer:
        raise ValueError("Model not compiled with optimizer")

    print("✓ Model setup validated successfully")


