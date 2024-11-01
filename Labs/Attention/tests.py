import tensorflow as tf
from helpers import LanguageIndex

def validate_data_loader(
    data_loader_func, 
    path, 
    batch_size, 
    samples=None, 
    max_len=None, 
    reverse=False
):
    """
    Validates the data_loader function to ensure that it returns the expected outputs.
    
    PARAMETERS
    ----------
    data_loader_func : function
        The data_loader function to validate.
    path : str
        Path to the translation file.
    batch_size : int
        The batch size for training.
    samples : int, optional
        Number of samples to load. If None, all samples are loaded.
    max_len : int, optional
        Maximum length of sequences. If None, the actual maximum length of the dataset is used.
    reverse : bool, optional
        Whether to reverse input-target pairs.
        
    Raises
    ------
    ValueError
        If any aspect of the data_loader output is incorrect.
    """
    try:
        # Load the data using the provided data_loader function
        train_dataset, test_dataset, inp_lang, tgt_lang, max_length_inp, max_length_tgt = data_loader_func(
            path, batch_size, max_len, reverse
        )
    except Exception as e:
        raise ValueError(f"Data loader failed to execute: {str(e)}")
    
    # Check if train_dataset and test_dataset are tf.data.Dataset instances
    if not isinstance(train_dataset, tf.data.Dataset):
        raise ValueError("train_dataset is not an instance of tf.data.Dataset")
    
    if not isinstance(test_dataset, tf.data.Dataset):
        raise ValueError("test_dataset is not an instance of tf.data.Dataset")
    
    # Check if inp_lang and tgt_lang are instances of LanguageIndex
    if not isinstance(inp_lang, LanguageIndex):
        raise ValueError("inp_lang is not an instance of LanguageIndex")
    
    if not isinstance(tgt_lang, LanguageIndex):
        raise ValueError("tgt_lang is not an instance of LanguageIndex")
    
    # Check if input vocabulary size and target vocabulary size are positive
    if len(inp_lang.word2idx) <= 0:
        raise ValueError("Input language vocabulary size must be greater than 0")
    
    if len(tgt_lang.word2idx) <= 0:
        raise ValueError("Target language vocabulary size must be greater than 0")
    
    # Check if max_length_inp and max_length_tgt are positive integers
    if max_length_inp <= 0:
        raise ValueError(f"max_length_inp must be positive, got {max_length_inp}")
    
    if max_length_tgt <= 0:
        raise ValueError(f"max_length_tgt must be positive, got {max_length_tgt}")
    
    # Validate that datasets have appropriate batch sizes
    train_batch_count = len(list(train_dataset))
    test_batch_count = len(list(test_dataset))
    
    if train_batch_count <= 0:
        raise ValueError("Training dataset has no batches, check the batch size or data loading process")
    
    if test_batch_count <= 0:
        raise ValueError("Test dataset has no batches, check the batch size or data loading process")

    # Validate individual data batches for input-output shapes
    for batch_inp, batch_tgt in train_dataset.take(1):
        if batch_inp.shape[1] != max_length_inp:
            raise ValueError(f"Input tensor shape mismatch. Expected max_length_inp {max_length_inp}, got {batch_inp.shape[1]}")
        
        if batch_tgt.shape[1] != max_length_tgt:
            raise ValueError(f"Target tensor shape mismatch. Expected max_length_tgt {max_length_tgt}, got {batch_tgt.shape[1]}")
    
    print("Data loader is working correctly.")



def validate_encoder(encoder, vocab_inp_size, embed_dim=64, units=50):
  """
  Checks if the encoder is properly defined by creating a test input 
  and verifying the output shapes.

  Args:
    encoder: The encoder instance to check.
    vocab_inp_size: The size of the input vocabulary.
    embed_dim: The embedding dimension. Defaults to 64.
    units: The number of units in the LSTM layer. Defaults to 50.

  Raises:
    ValueError: If the encoder output shapes do not match the expected shapes.
  """
  # Create a test input
  test_input = tf.constant([[1, 2, 3, 0, 0]])  # Example input sequence

  # Call the encoder
  enc_outputs, enc_state = encoder(test_input)

  # Check output shapes
  expected_enc_outputs_shape = (1, 5, units)  
  expected_enc_state_shape = (1, units)  

  if not enc_outputs.shape.as_list() == list(expected_enc_outputs_shape):
    raise ValueError(f"Encoder output shape mismatch. Expected: {expected_enc_outputs_shape}, Got: {enc_outputs.shape}")
  if not enc_state[-1].shape.as_list() == list(expected_enc_state_shape):
    raise ValueError(f"Encoder state shape mismatch. Expected: {expected_enc_state_shape}, Got: {enc_state[-1].shape}")

  print("Encoder checks passed!") # If no errors are raised, the check has passed



def validate_attention(attention):
    """
    Validates if an attention layer is defined appropriately.
    
    PARAMETERS
    ----------
    attention : object
        The attention object to validate. It should have attributes like `W_q`, `W_k`, and `W_v`.
    
    Raises
    ------
    ValueError
        If any attribute is missing or dimensions do not match.
    """
    required_attributes = ['W_q', 'W_k', 'W_v']

    # Check if the required attributes are present in the attention mechanism
    for attr in required_attributes:
        if not hasattr(attention, attr):
            raise ValueError(f"Attention mechanism is missing the required attribute '{attr}'")
    
    # Validate the units of the dense layers
    try:
        units = attention.W_q.units
        if units <= 0:
            raise ValueError(f"The number of units in W_q must be positive, got {units}")
        if attention.W_k.units != units:
            raise ValueError(f"W_k units ({attention.W_k.units}) do not match W_q units ({units})")
        if attention.W_v.units != 1:
            raise ValueError(f"W_v units should be 1 for scalar scores, got {attention.W_v.units}")
    except AttributeError as e:
        raise ValueError(f"Attention's dense layers are not defined properly: {str(e)}")
    
    print("Attention mechanism is defined correctly.")

def validate_decoder(decoder, vocab_tgt_size, embed_dim=64, units=50):
  """
  Checks if the decoder is properly defined by creating test inputs
  and verifying the output shapes.

  Args:
    decoder: The decoder instance to check.
    vocab_tgt_size: The size of the target vocabulary.
    embed_dim: The embedding dimension. Defaults to 64.
    units: The number of units in the LSTM layer. Defaults to 50.

  Raises:
    ValueError: If the decoder output shapes do not match the expected shapes.
  """
  # Create test inputs
  test_input = tf.constant([[1, 2, 0, 0]])  # Example target sequence
  test_enc_outputs = tf.random.normal((1, 5, units))  # Example encoder outputs
  test_enc_state = [tf.random.normal((1, units))]  # Example encoder state

  # Call the decoder
  dec_output, dec_state, att_wgts = decoder(
      test_input, test_enc_outputs, test_enc_state)

  # Check output shapes
  expected_dec_output_shape = (1, 4, vocab_tgt_size)
  expected_dec_state_shape = (1, units)
  expected_att_wgts_shape = (1, 4, 5)

  if not dec_output.shape.as_list() == list(expected_dec_output_shape):
    raise ValueError(
        f"Decoder output shape mismatch. Expected: {expected_dec_output_shape}, Got: {dec_output.shape}")
  if not dec_state[-1].shape.as_list() == list(expected_dec_state_shape):
    raise ValueError(
        f"Decoder state shape mismatch. Expected: {expected_dec_state_shape}, Got: {dec_state[-1].shape}")
  if not att_wgts.shape.as_list() == list(expected_att_wgts_shape):
    raise ValueError(
        f"Attention weights shape mismatch. Expected: {expected_att_wgts_shape}, Got: {att_wgts.shape}")

  print("Decoder checks passed!") # If no errors are raised, the check has passed

def test_model_loss(test_loss, min_val_loss=0.095):
  """Tests if the model achieved at least the minimum test loss.

  Args:
      test_accuracy: The accuracy of the model on the test data.
      min_val_loss: The minimum required validation loss.

  Raises:
      AssertionError if the minimum validation loss is not met.
  """
  assert test_loss <= min_val_loss, f"Model did not achieve minimum test loss of {min_val_loss:.2f}, achieved {test_loss:.2f}"
