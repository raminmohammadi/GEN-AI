import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Function to plot model metrics
def plot_metrics(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Function to plot the attention weights
def plot_attention_weights(attention, sentence, idx):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attention[idx], annot=True, cmap='viridis', ax=ax)
    # ax.matshow(attention[idx], cmap='viridis', )

    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))

    ax.set_xticklabels(sentence, rotation=90)
    ax.set_yticklabels(sentence)

    ax.set_xlabel('Attention for each word in input')
    ax.set_ylabel('Attention by each word in input')
    plt.show()

# Function to preprocess the input text
def preprocess_input_text(text):
    # Tokenize and convert the input text to sequences
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequence to ensure it has the same length as the training data
    padded_sequence = pad_sequences(sequences, maxlen=max_len)
    return padded_sequence
