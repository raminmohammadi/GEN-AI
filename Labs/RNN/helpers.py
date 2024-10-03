import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from tests import *

# DATA PREPROCESSING

def create_input_output_pairs(sequences, max_seq_length):
    """
    Creates input-output pairs from the tokenized sequences. The input will be 
    subsequences of the original sequence (up to max_seq_length), and the output 
    will be the next token in the sequence.
    
    Args:
        sequences (List[List[int]]): A list of tokenized sequences.
        max_seq_length (int): The maximum sequence length for truncation.

    Returns:
        np.array: Array of padded input sequences.
        np.array: Array of output words (next token in the sequence).
    """

    input_sequences = []
    output_words = []

    for seq in sequences:
        if len(seq) < 2:  # Skip sequences that are too short to generate pairs
            continue

        # Truncate the sequence if it's longer than max_seq_length
        truncated_seq = seq[:max_seq_length]

        # Generate input-output pairs
        for i in range(1, len(truncated_seq)):
            input_seq = truncated_seq[:i]  # Input is from start to the i-th token
            padded_input_seq = pad_sequences([input_seq], maxlen=max_seq_length, padding='pre')[0]
            output_word = truncated_seq[i]  # Output is the next token

            input_sequences.append(padded_input_seq)
            output_words.append(output_word)

    return np.array(input_sequences), np.array(output_words)

def truncate_sequences(sequences, max_seq_length):
    """
    Truncates each sequence in the list to a maximum sequence length. If a sequence
    exceeds the max length, it is truncated; otherwise, it is left as is.

    Args:
        sequences (List[List[int]]): A list of tokenized sequences.
        max_seq_length (int): Maximum allowed sequence length.

    Returns:
        List[List[int]]: A list of truncated sequences.
    """
    check_max_seq_length(max_seq_length)
    truncated_sequences = [seq[:max_seq_length] for seq in sequences]
    return truncated_sequences

# INFERENCE

# Function to generate predictions from the trained model
def generate_predictions(model, input_sequences, tokenizer):
    """
    Generates text predictions by sequentially predicting the next word in a sequence.

    Args:
        model (keras.Model): The trained RNN model.
        tokenizer (Tokenizer): The tokenizer used for text processing.
        seed_text (str): Initial input text to start generating predictions.
        max_seq_length (int): Maximum sequence length for padding.
        num_words (int): Number of words to generate.

    Returns:
        str: The generated sequence of words.
    """

    predictions = []
    for input_seq in input_sequences:
        # Reshape input for prediction (batch_size=1)
        input_seq = np.expand_dims(input_seq, axis=0)

        # Predict the next token (the output is a probability distribution)
        predicted_probs = model.predict(input_seq, verbose=0)

        # Take the token with the highest probability as the prediction
        predicted_token = np.argmax(predicted_probs, axis=-1)

        # Convert the predicted token index back to a word
        predicted_word = tokenizer.index_word.get(predicted_token[0], "<UNK>")
        predictions.append(predicted_word)
    return predictions

# Convert the true output tokens back to words
def convert_token_ids_to_words(token_ids, tokenizer):
    """
    Converts a list of token IDs back to words using the tokenizer's word index.

    Args:
        token_ids (List[int]): A list of token IDs (predicted or actual).
        tokenizer (Tokenizer): The tokenizer used for text processing.

    Returns:
        str: A sequence of words corresponding to the input token IDs.
    """
    return [tokenizer.index_word.get(token_id, "<UNK>") for token_id in token_ids]

# Function to calculate the BLEU score
def calculate_bleu(predictions, references):
    """
    Calculates the BLEU score to evaluate the quality of text generation by comparing 
    the predicted sequences to the reference sequences.

    Args:
        references (List[str]): List of ground truth text sequences.
        predictions (List[str]): List of predicted text sequences.

    Returns:
        float: BLEU score (a value between 0 and 1, where 1 indicates a perfect match).
    """
    references = [[ref] for ref in references]  # references need to be wrapped in an extra list

    # Compute corpus-level BLEU score
    bleu_score = corpus_bleu(references, predictions)
    print(f'BLEU Score: {bleu_score:.2f}')
    print('\n')
    test_bleu_score(bleu_score)

    return bleu_score