import matplotlib.pyplot as plt

def plot_denoised_images(model, x_train, x_train_noisy, num_images=10):
  """
  Plots a grid of original, noisy, and denoised images using a trained VAE model.

  Args:
      model: The trained VAE model.
      x_train: The original, clean training data (assumed to be reshaped to 2D).
      x_train_noisy: The noisy training data (assumed to be reshaped to 2D).
      num_images: Number of images to display in the grid (default: 10).
  """

  denoised_images = model.predict(x_train_noisy[:num_images])  # Get predictions

  img_width = 28  
  img_height = 28 

  rows, cols = 3, (num_images // 3) + (num_images % 3 > 0)  # Ensure enough columns

  fig, axs = plt.subplots(rows, cols, figsize=(15, 6))

  for i in range(num_images):
    original_image = x_train[i].reshape(img_width, img_height)
    noisy_image = x_train_noisy[i].reshape(img_width, img_height)
    denoised_image = denoised_images[i].reshape(img_width, img_height)

    axs[0, i % cols].imshow(original_image, cmap='gray')
    axs[0, i % cols].set_title("Original")
    axs[1, i % cols].imshow(noisy_image, cmap='gray')
    axs[1, i % cols].set_title("Noisy")
    axs[2, i % cols].imshow(denoised_image, cmap='gray')
    axs[2, i % cols].set_title("Denoised")

  fig.tight_layout()
  plt.show()