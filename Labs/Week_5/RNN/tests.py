from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, BatchNormalization


def check_max_seq_length(max_seq_length, required_length=500):
    """
    Checks whether the maximum sequence length is set to at least the required length.
    If not, raises a ValueError.

    Args:
        max_seq_length (int): The current maximum sequence length.
        required_length (int, optional): The required minimum length. Defaults to 5000.

    Raises:
        ValueError: If max_seq_length is less than the required_length.
    """
    if max_seq_length < required_length:
        raise ValueError(f"Max sequence length is too short: {max_seq_length} (< {required_length}). "
                         f"Please set max_seq_length to at least {required_length}.")
    
def test_model_structure(model, vocab_size):
    """
    Checks if the model structure satisfies the following conditions:
    1. At least one SimpleRNN layer is present.
    2. At least one hidden layer is present.
    3. The output layer has the same number of units as the vocab_size.

    Args:
        model (keras.Model): The Keras model to be checked.
        vocab_size (int): The size of the vocabulary (number of unique tokens).

    Raises:
        ValueError: If any of the conditions are not met.
    """
    # Check for at least one SimpleRNN layer
    rnn_layers = [layer for layer in model.layers if isinstance(layer, SimpleRNN)]
    if not rnn_layers:
        raise ValueError("The model must have at least one SimpleRNN layer.")
    
    # Check for at least one hidden layer (RNN or Dense)
    hidden_layers = [layer for layer in model.layers if isinstance(layer, SimpleRNN) or isinstance(layer, Dense)]
    if len(hidden_layers) < 2:  # Hidden layers include RNN or Dense, excluding input and output
        raise ValueError("The model must have at least one hidden layer (e.g., SimpleRNN or Dense).")
    
    # Check if the output layer's units match vocab_size
    output_layer = model.layers[-1]
    if not isinstance(output_layer, Dense) or output_layer.units != vocab_size:
        raise ValueError(f"The output layer must have {vocab_size} units (matching the vocab size), "
                         f"but it has {output_layer.units} units.")


def test_validation_accuracy(history, min_val_acc=0.25):
    """
    Checks whether the validation accuracy is at least the specified threshold.
    If not, raises a ValueError.

    Args:
        val_acc (float): The validation accuracy to be checked.
        min_val_acc (float, optional): The minimum acceptable validation accuracy. Defaults to 0.25.

    Raises:
        ValueError: If the validation accuracy is below the threshold.
    """
    val_acc = max(history.history['val_accuracy'])
    if val_acc < min_val_acc:
        raise ValueError(f"Validation accuracy is too low: {val_acc:.2f} (< {min_val_acc}). "
                         f"Please improve the model performance.")
    else:
        print(f"Validation accuracy is sufficient: {val_acc:.2f} (>= {min_val_acc}).")


def test_bleu_score(score, min_bleu_score=0.15):
    """
    Checks whether the BLEU score achieved is atleast the specified threshold.
    If not, raises a ValueError.

    Args:
        score (float): The bleu score to be checked.
        min_bleu_score (float, optional): The minimum acceptable validation accuracy. Defaults to 0.15.

    Raises:
        ValueError: If the bleu score is below the threshold.

    """
    if score < min_bleu_score:
        raise ValueError(f"BLEU Score achieved is too low: {score:.2f} (< {min_bleu_score})."
                         f"Please improve the model performance.")

    else:
            print(f"Sufficient BLEU Score achieved: {score:.2f} (>= {min_bleu_score}).")
