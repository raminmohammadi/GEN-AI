import platform
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Detects and sets the device to be used for training
def detect_and_set_device():
    if tf.test.is_built_with_cuda() or platform.system() == "Darwin":
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            print("GPU is available. Using GPU.")
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return 'GPU'
            except RuntimeError as e:
                print(f"Unable to set memory growth: {e}")
                return 'CPU'
    print("GPU is not available. Using CPU.")
    return 'CPU'

def load_data():
    print("Downloading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    train_data = pd.DataFrame(dataset['train'])
    test_data = pd.DataFrame(dataset['test'])
    
    # Rename columns to match our desired structure
    train_data = train_data.rename(columns={'text': 'Text', 'label': 'Class'})
    test_data = test_data.rename(columns={'text': 'Text', 'label': 'Class'})
    
    # 'Class' is already an integer, but it's 0-indexed. Add 1 to make it 1-indexed
    train_data['Class'] = train_data['Class'] + 1
    test_data['Class'] = test_data['Class'] + 1
    
    # Ensure 'Text' is string
    train_data['Text'] = train_data['Text'].astype(str)
    test_data['Text'] = test_data['Text'].astype(str)
    
    print("Dataset loaded successfully.")
    return train_data, test_data


# 7. Visualization Function
def plot_results(history, test_data, predicted_classes):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(test_data['Class'] - 1, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.show()