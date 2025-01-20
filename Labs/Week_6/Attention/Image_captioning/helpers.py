import concurrent.futures
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import string
import time
import urllib.request

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tqdm
from helpers import *
import tensorflow as tf
import tensorflow_text as text
import tensorflow_datasets as tfds

IMAGE_SHAPE=(224, 224, 3)


def flickr8k(path='flickr8k'):
    # Set the data path
    path = pathlib.Path(path)

    # Check if the necessary files are already present, if not download them
    if len(list(path.rglob('*'))) < 16197:
        # Download the Flickr8k images dataset
        tf.keras.utils.get_file(
            origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
            cache_dir='.',
            cache_subdir=path,
            extract=True)
        # Download the text data for Flickr8k
        tf.keras.utils.get_file(
            origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
            cache_dir='.',
            cache_subdir=path,
            extract=True)

    # Load and process the captions
    captions = (path/"Flickr8k_text.zip"/"Flickr8k.token.txt").read_text().splitlines()
    captions = (line.split('\t') for line in captions)
    captions = ((fname.split('#')[0], caption) for (fname, caption) in captions)

    # Organize captions into a dictionary where each key is a filename
    cap_dict = collections.defaultdict(list)
    for fname, cap in captions:
        cap_dict[fname].append(cap)

    # Load the training image filenames and map captions to them
    train_files = (path/"Flickr8k_text.zip"/'Flickr_8k.trainImages.txt').read_text().splitlines()
    train_captions = [(str(path/"Flickr8k_Dataset.zip"/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in train_files]

    # Load the test image filenames and map captions to them
    test_files = (path/"Flickr8k_text.zip"/'Flickr_8k.testImages.txt').read_text().splitlines()
    test_captions = [(str(path/"Flickr8k_Dataset.zip"/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in test_files]

    # Create TensorFlow datasets for the training and test sets
    train_ds = tf.data.experimental.from_list(train_captions)
    test_ds = tf.data.experimental.from_list(test_captions)

    return train_ds, test_ds


def conceptual_captions(*, data_dir="conceptual_captions", num_train, num_val):
    # Function to iterate through the index file and yield captions and URLs
    def iter_index(index_path):
        with open(index_path) as f:
            for line in f:
                caption, url = line.strip().split('\t')
                yield caption, url

    # Function to download images from URLs and save them locally
    def download_image_urls(data_dir, urls):
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        def save_image(url):
            hash = hashlib.sha1(url.encode())
            file_path = data_dir / f'{hash.hexdigest()}.jpeg'
            if file_path.exists():
                return file_path  # Skip download if already exists

            try:
                result = requests.get(url, timeout=5)  # Attempt to download the image
            except Exception:
                file_path = None
            else:
                file_path.write_bytes(result.content)
            return file_path

        result = []
        out_paths = ex.map(save_image, urls)  # Download images concurrently
        for file_path in tqdm.tqdm(out_paths, total=len(urls)):
            result.append(file_path)

        return result

    # Function to create a TensorFlow dataset from the index file
    def ds_from_index_file(index_path, data_dir, count):
        data_dir.mkdir(exist_ok=True)
        index = list(itertools.islice(iter_index(index_path), count))
        captions = [caption for caption, url in index]
        urls = [url for caption, url in index]

        paths = download_image_urls(data_dir, urls)

        new_captions = []
        new_paths = []
        for cap, path in zip(captions, paths):
            if path is None:
                continue  # Skip if download failed
            new_captions.append(cap)
            new_paths.append(path)

        new_paths = [str(p) for p in new_paths]

        ds = tf.data.Dataset.from_tensor_slices((new_paths, new_captions))
        ds = ds.map(lambda path, cap: (path, cap[tf.newaxis]))  # Adjust dataset structure
        return ds

    data_dir = pathlib.Path(data_dir)
    train_index_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv',
        cache_subdir=data_dir / 'train',
        cache_dir='.')

    val_index_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv',
        cache_subdir=data_dir / 'val',
        cache_dir='.')

    # Create training and validation datasets
    train_raw = ds_from_index_file(train_index_path, data_dir=data_dir / 'train', count=num_train)
    test_raw = ds_from_index_file(val_index_path, data_dir=data_dir / 'val', count=num_val)

    return train_raw, test_raw


def load_image(image_path):
    """
    Loads an image from a file path, decodes it, and resizes it to a predefined shape.

    Args:
    image_path (str): Path to the image file.

    Returns:
    tf.Tensor: The processed image tensor.
    """
    # Read the image file from the specified path
    img = tf.io.read_file(image_path)
    
    # Decode the image file from JPEG format, assuming it is in JPEG format
    img = tf.io.decode_jpeg(img, channels=3)  # Use channels=3 to ensure the image has three color channels (RGB)
    
    # Resize the image to the specified dimensions (299x299 in this example) without the batch dimension
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])  # Resize the image to the target shape
    
    return img

def detokenize(tokens, idx2word):
    text = ""
    for t in tokens:
        if 'tensorflow' in str(type(tokens)):
            text += idx2word[t.numpy()] + ' '
        else:
            text += idx2word[t] + ' '
    text = text.replace(' <pad>', '')
    text = text.replace('<start>', '')
    text = text.replace('<end>', '')
    return text.strip()

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
def plot_attention_weights(attention, sentence, idx = None):
    if idx:
        attention = attention[idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(attention, annot=True, cmap='viridis', ax=ax)
    # ax.matshow(attention[idx], cmap='viridis', )

    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))

    ax.set_xticklabels(sentence, rotation=90)
    ax.set_yticklabels(sentence)

    ax.set_xlabel('Attention for each word in input')
    ax.set_ylabel('Attention by each word in input')
    plt.show()


def match_shapes(images, captions):
    """
    Adjusts the shapes of image feature maps and captions to align them for batching.

    Args:
    images (tf.Tensor): Tensor of image feature maps with shape [batch_size, ...].
    captions (tf.Tensor): Tensor of captions with shape [batch_size, num_captions].

    Returns:
    Tuple[tf.Tensor, tf.Tensor]: A tuple containing reshaped images and captions.
    """
    # Parse the shape of captions to get the number of captions per image (batch, captions)
    caption_shape = einops.parse_shape(captions, 'b c')
    
    # Rearrange captions to a flat list from batched groups of captions
    captions = einops.rearrange(captions, 'b c -> (b c)')
    
    # Repeat each image tensor for each of its corresponding captions
    images = einops.repeat(
        images, 'b ... -> (b c) ...',
        c=caption_shape['c'])
    
    return images, captions


def standardize(s):
    """
    Standardizes text by converting it to lowercase, removing punctuation,
    and adding start and end tokens.

    Args:
    s (tf.Tensor): A string tensor that contains the text to be standardized.

    Returns:
    tf.Tensor: The standardized text as a string tensor.
    """
    # Convert all characters in the input string tensor to lowercase
    s = tf.strings.lower(s)
    
    # Remove all punctuation from the string using a regular expression
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    
    # Add start and end tokens to the string to denote the beginning and end of the text
    s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
    
    return s


def save_dataset(ds, save_path, image_model, tokenizer, shards=10, batch_size=32):
    """
    Processes and saves a dataset in sharded files.

    Args:
    ds (tf.data.Dataset): The input dataset containing paths and captions.
    save_path (str): The directory to save the processed dataset.
    image_model (Model): The model to extract image features.
    tokenizer (Tokenizer): The tokenizer for processing captions.
    shards (int): Number of shards to split the dataset into.
    batch_size (int): Number of items per batch.

    """
    # Prepare the dataset by loading images, applying the image model, and batching
    ds = (ds
          .map(lambda path, caption: (load_image(path), caption))  # Load and process each image
          .apply(tf.data.experimental.ignore_errors())  # Ignore errors in data processing
          .batch(batch_size))  # Batch the data

    # Generator to apply the image model on the CPU to avoid potential issues on GPU
    def gen():
        for (images, captions) in tqdm.tqdm(ds):
            feature_maps = image_model(images)
            feature_maps, captions = match_shapes(feature_maps, captions)
            yield feature_maps, captions

    # Wrap the generator with the appropriate output types
    new_ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=image_model.output_shape),
            tf.TensorSpec(shape=(None,), dtype=tf.string)))

    # Process text and shuffle data
    new_ds = (new_ds
              .map(lambda imgs, txts: prepare_txt(imgs, txts, tokenizer=tokenizer), num_parallel_calls=tf.data.AUTOTUNE)
              .unbatch()  # Flatten the dataset
              .shuffle(1000))  # Shuffle the dataset

    # Function to determine the shard of each data item
    def shard_func(i, item):
        return i % shards

    # Save the dataset in sharded files
    new_ds.enumerate().save(save_path, shard_func=shard_func)

def load_dataset(save_path, batch_size=32, shuffle=1000, cycle_length=2):
    """
    Load a sharded dataset from disk.

    Args:
    save_path (str): Path where the dataset is saved.
    batch_size (int): Number of items per batch.
    shuffle (int): Buffer size for shuffling.
    cycle_length (int): Number of datasets to interleave.

    """
    # Custom function to handle reading and interleaving multiple dataset files
    def custom_reader_func(datasets):
        datasets = datasets.shuffle(shuffle)
        return datasets.interleave(lambda x: x, cycle_length=cycle_length)

    # Load the dataset and apply the custom reader function
    ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)

    # Function to remove the enumeration index from the dataset
    def drop_index(i, x):
        return x

    # Final dataset processing steps
    ds = (ds
          .map(drop_index, tf.data.AUTOTUNE)  # Remove indexes added during save
          .shuffle(shuffle)  # Shuffle the dataset
          .padded_batch(batch_size)  # Pad and batch the dataset
          .prefetch(tf.data.AUTOTUNE))  # Prefetch data for faster access

    return ds


def prepare_txt(imgs, txts, tokenizer):
    """
    Prepares text data for training by tokenizing and structuring input and label tokens.

    Args:
    imgs (tf.Tensor): A tensor of image data associated with the texts.
    txts (tf.Tensor): A tensor containing text data to be processed.

    Returns:
    Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]: A tuple where the first element is
    a tuple of images and input tokens, and the second element is the label tokens.
    """
    # Tokenize the text data into integer tokens using a provided tokenizer
    tokens = tokenizer(txts)

    # Create input tokens by excluding the last token in each sequence for training
    input_tokens = tokens[..., :-1]
    
    # Create label tokens by excluding the first token in each sequence to use as targets for training
    label_tokens = tokens[..., 1:]
    
    # Cast label tokens to an integer type suitable for model training (int32 is commonly used)
    label_tokens = tf.cast(label_tokens, dtype=tf.int32)  # or change to tf.int64 if needed based on the model's requirement

    # Return the processed tensors structured for input into a model
    return (imgs, input_tokens), label_tokens


def prepare_dataset(ds, tokenizer, batch_size=32, shuffle_buffer=1000):
    """
    Prepares a dataset for training by processing images and text, shuffling, batching, and tokenizing.

    Args:
    ds (tf.data.Dataset): The dataset containing image paths and associated captions.
    tokenizer (Tokenizer): A tokenizer instance used to convert text to token sequences.
    batch_size (int): Number of elements in each batch.
    shuffle_buffer (int): Size of the buffer used for shuffling the data.

    Returns:
    tf.data.Dataset: The processed dataset ready for training.
    """
    # Shuffle the dataset and load images, and apply batching
    ds = (ds
          .shuffle(10000)  # Shuffle the dataset with a buffer of 10,000 items
          .map(lambda path, caption: (load_image(path), caption), num_parallel_calls=tf.data.AUTOTUNE)  # Load images and keep captions
          .apply(tf.data.experimental.ignore_errors())  # Ignore any errors during processing
          .batch(batch_size))  # Batch the data

    # Define a function to convert datasets to tensors
    def to_tensor(inputs, labels):
        (images, in_tok), out_tok = inputs, labels
        # Convert input tokens and output tokens to tensors
        out_tok = tf.cast(out_tok, dtype=tf.int32)  # Cast output tokens to int32 for training compatibility
        return (images, in_tok.to_tensor()), out_tok.to_tensor()  # Convert in_tok to tensor if needed

    # Return the fully prepared dataset
    return (ds
            .map(match_shapes, num_parallel_calls=tf.data.AUTOTUNE)  # Adjust shapes of images and captions
            .unbatch()  # Unbatch the dataset to flatten it
            .shuffle(shuffle_buffer)  # Shuffle the dataset with specified buffer
            .batch(batch_size)  # Re-batch the dataset after shuffling
            .map(lambda imgs, txts: prepare_txt(imgs, txts, tokenizer=tokenizer), num_parallel_calls=tf.data.AUTOTUNE)
            .map(to_tensor, num_parallel_calls=tf.data.AUTOTUNE))  # Convert processed data to tensors
