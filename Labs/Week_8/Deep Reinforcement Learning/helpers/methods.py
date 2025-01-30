import platform
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

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


def plot_training_metrics(agent, rewards_history, epsilon_history, save_path='output/training_metrics.png'):
    """Plot training metrics"""
    
    # Ensure the directory exists before saving the plot
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Output directory '{save_dir}' created.")
    
    # Plot rewards
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot rewards
    axes[0, 0].plot(rewards_history)
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')

    # Plot epsilon decay
    axes[0, 1].plot(epsilon_history)
    axes[0, 1].set_title('Epsilon Decay')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon Value')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    
    
def save_frames_as_gif(frames, filename='training.gif', fps=30):
    """
    Save frames as GIF with proper frame formatting
    """
    try:
        # Convert frames to numpy arrays
        processed_frames = []
        for frame in frames:
            # Convert frame to numpy array if it isn't already
            frame = np.array(frame)

            # Ensure frame is in RGB format
            if frame.ndim == 3 and frame.shape[-1] == 3:
                # Convert to uint8 if needed
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                processed_frames.append(frame)

        # Save as GIF if we have valid frames
        if processed_frames:
            imageio.mimsave(filename, processed_frames, fps=fps)
            print(f"Successfully saved GIF to {filename}")
        else:
            print("No valid frames to save")

    except Exception as e:
        print(f"Error saving GIF: {e}")