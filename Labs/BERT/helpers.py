import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, TFDistilBertForQuestionAnswering, create_optimizer
import tensorflow as tf
import seaborn as sns

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

def plot_training_history(history):
    """
    Creates multiple plots to visualize the training history:
    1. Training & Validation Loss
    2. Start & End Position Losses
    3. Combined metrics visualization
    
    Args:
        history: History object returned by model.fit()
    """

    
    # Set the style for better visualization
    plt.style.use('seaborn')
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Overall Loss Plot
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Start Position Loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['start_positions_loss'], 
             label='Start Position Training Loss')
    plt.plot(history.history['val_start_positions_loss'], 
             label='Start Position Validation Loss')
    plt.title('Start Position Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 3. End Position Loss
    plt.subplot(2, 2, 3)
    plt.plot(history.history['end_positions_loss'], 
             label='End Position Training Loss')
    plt.plot(history.history['val_end_positions_loss'], 
             label='End Position Validation Loss')
    plt.title('End Position Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 4. Combined Metrics
    plt.subplot(2, 2, 4)
    metrics = ['loss', 'start_positions_loss', 'end_positions_loss']
    for metric in metrics:
        plt.plot(history.history[metric], label=f'Training {metric}')
    plt.title('Combined Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    


def get_answer(question, context, model, tokenizer):
    inputs = tokenizer(question, context, return_tensors='tf')
    outputs = model(**inputs)

    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0]

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1])
    )

    return answer


# Preprocessing function
def preprocess_function_(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # TODO: Implement tokenization using the BERT tokenizer
    # Step 1: Use the tokenizer to tokenize questions and contexts
    # - Set truncation to "only_second" to ensure the context is truncated, not the question.
    # - Set max_length to 384 to ensure tokenized input fits the maximum BERT length.
    # - Set stride to 128 for overlapping tokens when truncating
    # - Use padding to ensure all sequences in a batch are the same length
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # TODO: Extract the position of answers
    # Step 1: Initialize lists to hold start and end positions
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    # Step 2: Iterate through each offset to find the token indices matching the start and end of answers
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = sequence_ids.index(1, context_start + 1) - 1

        # Step 3: Check if the entire answer is present in the context
        # If the answer is outside the present context, label it appropriately (e.g., (0, 0))
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Step 4: Find the tokens that correspond to the start and end positions of the answer
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    # Step 5: Add the start and end positions to the model inputs
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def create_qa_model_():
    # Step 1: Load the pre-trained BERT model for question answering from Hugging Face
    model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    # Step 2: Create an optimizer for training
    num_train_steps = 1000  # This should be the total number of training steps
    # - init_lr: The initial learning rate
    # - num_warmup_steps: Gradually increase the learning rate to the target value
    optimizer, lr_schedule = create_optimizer(
        init_lr=5e-5,
        num_warmup_steps=0,
        num_train_steps=num_train_steps
    )

    # Step 3: Define the loss function for the model, ensuring it's suitable for the question-answering task
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Step 4: Compile the model with the optimizer and the loss function
    model.compile(optimizer=optimizer, loss=[loss, loss])

    return model

