import matplotlib.pyplot as plt
import numpy
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


def run_inference(model, text_cleaner, tokenizer, sentence, labels):
    """Runs inference on a single sentence using the trained model.

    Args:
        model: The trained multi-head attention model.
        tokenizer: The tokenizer used to tokenize the input sentence.
        sentence: The input sentence to run inference on.
        labels: A list of labels corresponding to the model's output classes.

    Returns:
        The predicted sentiment label for the input sentence.
    """

    # 1. Clean the input sentence:
    cleaned_sentence = text_cleaner(sentence)  

    # 2. Tokenize the cleaned sentence:
    sequence = tokenizer.texts_to_sequences([cleaned_sentence])  

    # 3. Pad the sequence:
    padded_sequence = pad_sequences(sequence, padding='post')  

    # 4. Make prediction using the model:
    prediction = model.predict(padded_sequence)  

    # 5. Get the predicted label:
    predicted_label_index = numpy.argmax(prediction) 
    predicted_label = labels[predicted_label_index]

    return predicted_label

