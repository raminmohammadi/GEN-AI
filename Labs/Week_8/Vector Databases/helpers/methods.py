import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from dataclasses import dataclass
import os
import requests
import zipfile
from PIL import Image

# Configuration
@dataclass
class Config:
    # Dataset configuration
    data_root: str = 'data'
    flickr8k_path: str = 'data/Flickr8k'
    image_dir: str = 'data/Flickr8k/Flicker8k_Dataset'
    caption_file: str = 'data/Flickr8k/Flickr8k.token.txt'
    train_images_file: str = 'data/Flickr8k/Flickr_8k.trainImages.txt'
    test_images_file: str = 'data/Flickr8k/Flickr_8k.testImages.txt'
    
    # Model configuration
    image_size: tuple = (224, 224)
    batch_size: int = 32


def download_dataset(config: Config):
    """Download and extract the Flickr8k dataset"""
    
    # Updated reliable download URLs
    FLICKR_URLS = {
        'images': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
        'text': 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
    }

    def download_file(url: str, filename: str):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            print(f"\nDownloading {filename}...")
            progress_bar = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024
            )

            with open(filename, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                raise Exception("Downloaded size does not match expected size")
            
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)
            return False

    def extract_zip(filename: str, extract_path: str):
        """Extract zip file"""
        try:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                print(f"\nExtracting {filename}...")
                zip_ref.extractall(extract_path)
            return True
        except Exception as e:
            print(f"Error extracting {filename}: {str(e)}")
            return False

    # Create necessary directories
    os.makedirs(config.data_root, exist_ok=True)
    os.makedirs(config.flickr8k_path, exist_ok=True)

    # Download and extract dataset files
    try:
        # Download and extract images
        image_zip = os.path.join(config.data_root, "Flickr8k_Dataset.zip")
        if not os.path.exists(config.image_dir):
            if download_file(FLICKR_URLS['images'], image_zip):
                if extract_zip(image_zip, config.flickr8k_path):
                    print("Images downloaded and extracted successfully")
                os.remove(image_zip)
            else:
                raise Exception("Failed to download or extract images")

        # Download and extract text files
        text_zip = os.path.join(config.data_root, "Flickr8k_text.zip")
        if not os.path.exists(config.caption_file):
            if download_file(FLICKR_URLS['text'], text_zip):
                if extract_zip(text_zip, config.flickr8k_path):
                    print("Text files downloaded and extracted successfully")
                os.remove(text_zip)
            else:
                raise Exception("Failed to download or extract text files")

        print("\nDataset download and extraction completed successfully!")
        
    except Exception as e:
        print(f"\nError during dataset download: {str(e)}")
        print("\nPlease try:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space")
        print("3. Verify you have write permissions in the target directory")
        raise

# Update the verify_dataset function
def verify_dataset(config: Config):
    """Verify the dataset was downloaded correctly"""
    try:
        # Check if directories exist
        if not os.path.exists(config.image_dir):
            raise FileNotFoundError(f"Image directory not found at {config.image_dir}")
        
        if not os.path.exists(config.caption_file):
            raise FileNotFoundError(f"Caption file not found at {config.caption_file}")
        
        # Count images
        image_count = len([f for f in os.listdir(config.image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Count captions
        with open(config.caption_file, 'r', encoding='utf-8') as f:
            caption_count = sum(1 for line in f)
        
        print(f"\nDataset verification:")
        print(f"Found {image_count} images in {config.image_dir}")
        print(f"Found {caption_count} captions in {config.caption_file}")
        
        # Basic validation
        if image_count < 8000:
            raise ValueError(f"Expected at least 8000 images, but found only {image_count}")
        
        if caption_count < 40000:
            raise ValueError(f"Expected at least 40000 captions, but found only {caption_count}")
        
        return True
        
    except Exception as e:
        print(f"\nDataset verification failed: {str(e)}")
        return False
    
    
def visualize_sample_images(config: Config, df: pd.DataFrame, num_samples=5):
    """Visualize random images with their captions"""
    try:
        # Get unique images
        unique_images = df['image'].unique()
        np.random.shuffle(unique_images)
        samples = unique_images[:num_samples]
        
        plt.figure(figsize=(20, 4))
        
        for idx, img_name in enumerate(samples):
            img_path = os.path.join(config.image_dir, img_name)
            
            try:
                img = Image.open(img_path)
                captions = df[df['image'] == img_name]['caption'].values
                
                plt.subplot(1, num_samples, idx+1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Image: {img_name}\nCaption: {captions[0][:50]}...", 
                         fontsize=10, pad=10)
            except Exception as img_error:
                print(f"Error loading image {img_name}: {str(img_error)}")
                continue
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        
        
def visualize_sample_images(config: Config, df: pd.DataFrame, num_samples=5):
    """Visualize random images with their captions"""
    try:
        # Get unique images
        unique_images = df['image'].unique()
        np.random.shuffle(unique_images)
        samples = unique_images[:num_samples]
        
        plt.figure(figsize=(20, 4))
        
        for idx, img_name in enumerate(samples):
            img_path = os.path.join(config.image_dir, img_name)
            
            try:
                img = Image.open(img_path)
                captions = df[df['image'] == img_name]['caption'].values
                
                plt.subplot(1, num_samples, idx+1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Image: {img_name}\nCaption: {captions[0][:50]}...", 
                         fontsize=10, pad=10)
            except Exception as img_error:
                print(f"Error loading image {img_name}: {str(img_error)}")
                continue
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        
        
def visualize_feature_distribution(features):
    """
    Visualize the distribution of extracted features
    
    Args:
        features: np.array of extracted features
    """
    plt.figure(figsize=(15, 5))
    
    # Plot feature means distribution
    plt.subplot(1, 2, 1)
    sns.histplot(features.mean(axis=0), bins=50)
    plt.title('Distribution of Feature Means')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    
    # Plot feature correlation matrix
    plt.subplot(1, 2, 2)
    correlation_matrix = np.corrcoef(features[:50].T)[:50, :50]
    sns.heatmap(correlation_matrix,
                cmap='coolwarm',
                xticklabels=10,
                yticklabels=10)
    plt.title('Feature Correlation Matrix\n(First 50 dimensions)')
    
    plt.tight_layout()
    plt.show()
    
    
def fetch_random_vectors(config, num_samples=5):
    """Fetch random vectors from Pinecone"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=config.pinecone_api_key)
        index = pc.Index(config.index_name)

        # Get all vector IDs from the index
        stats = index.describe_index_stats()

        # Get list of all vectors
        query_vector = np.zeros(2048)  # Zero vector for querying
        results = index.query(
            vector=query_vector.tolist(),
            top_k=num_samples,
            include_metadata=True
        )

        return results.matches
    except Exception as e:
        print(f"Error fetching vectors: {str(e)}")
        return None

def load_original_image(image_id, config):
    """Load original image from the dataset"""
    try:
        img_path = os.path.join(config.image_dir, image_id)
        return Image.open(img_path)
    except Exception as e:
        print(f"Error loading image {image_id}: {str(e)}")
        return None

def display_images_and_captions(matches, config):
    """Display original images with captions"""
    if not matches:
        print("No matches found")
        return None

    # Create figure
    plt.figure(figsize=(20, 4))

    # Create DataFrame to store information
    data = []

    for idx, match in enumerate(matches):
        image_id = match.id
        metadata = match.metadata

        # Load original image
        img = load_original_image(image_id, config)

        if img:
            # Display image
            plt.subplot(1, len(matches), idx+1)
            plt.imshow(img)
            plt.title(f"Image: {image_id}\nCaption: {metadata.get('captions', [''])[0][:50]}...")
            plt.axis('off')

            # Add to DataFrame
            data.append({
                'Image_ID': image_id,
                'Caption': metadata.get('captions', ['No caption'])[0],
            })

    plt.tight_layout()
    plt.show()

    # Create and display DataFrame
    df = pd.DataFrame(data)
    return df


def visualize_pinecone_results(config):
    # Fetch random vectors from Pinecone
    matches = fetch_random_vectors(config, num_samples=5)

    if matches:
        # Display images and get DataFrame
        df = display_images_and_captions(matches, config)
        print("\nImage Information:")
        print(df)
    else:
        print("Failed to fetch vectors from Pinecone")