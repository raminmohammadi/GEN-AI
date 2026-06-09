from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, BatchNormalization
import tensorflow as tf

def check_max_seq_length(max_seq_length, required_length=None):
    if required_length is None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            required_length = 500
            print(f"GPU detected. Required max_seq_length: {required_length}")
        else:
            required_length = 200
            print(f"No GPU detected (CPU mode). Required max_seq_length: {required_length}")

    if max_seq_length < required_length:
        raise ValueError(f"Max sequence length is too short: {max_seq_length} (< {required_length}). "
                         f"Please set max_seq_length to at least {required_length}.")
    
def test_model_structure(model, vocab_size):
    rnn_layers = [layer for layer in model.layers if isinstance(layer, SimpleRNN)]
    if not rnn_layers:
        raise ValueError("The model must have at least one SimpleRNN layer.")
    hidden_layers = [layer for layer in model.layers if isinstance(layer, SimpleRNN) or isinstance(layer, Dense)]
    if len(hidden_layers) < 2:
        raise ValueError("The model must have at least one hidden layer (e.g., SimpleRNN or Dense).")
    output_layer = model.layers[-1]
    if not isinstance(output_layer, Dense) or output_layer.units != vocab_size:
        raise ValueError(f"The output layer must have {vocab_size} units (matching the vocab size), "
                         f"but it has {output_layer.units} units.")

def test_validation_accuracy(history, min_val_acc=0.25):
    val_acc = max(history.history['val_accuracy'])
    if val_acc < min_val_acc:
        raise ValueError(f"Validation accuracy is too low: {val_acc:.2f} (< {min_val_acc}). "
                         f"Please improve the model performance.")
    else:
        print(f"Validation accuracy is sufficient: {val_acc:.2f} (>= {min_val_acc}).")

def test_bleu_score(score, min_bleu_score=0.15):
    if score < min_bleu_score:
        raise ValueError(f"BLEU Score achieved is too low: {score:.2f} (< {min_bleu_score})."
                         f"Please improve the model performance.")
    else:
        print(f"Sufficient BLEU Score achieved: {score:.2f} (>= {min_bleu_score}).")
