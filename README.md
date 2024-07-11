# CelebA GAN Face Generator

This repository contains a Generative Adversarial Network (GAN) implementation for generating celebrity-like faces using the CelebA dataset. The project includes a main script for training the GAN and generating images, as well as a separate GIF generator script.

## Features

- GAN architecture for generating realistic face images
- Training on the CelebA dataset
- Customizable network architecture and hyperparameters
- Periodic saving of generated images during training
- GIF generator for creating animations of the training progress

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- Pillow

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/celeba-gan-face-generator.git
   cd celeba-gan-face-generator
   ```

2. Install the required packages:
   ```
   pip install torch torchvision matplotlib numpy Pillow
   ```

3. Download the CelebA dataset and place it in the `./data/celeba` directory. The script expects a zip file named `img_align_celeba.zip` in this location.

## Configuration

The script uses a `CONFIG` dictionary for easy customization. Key configurations include:

- `latent_dim`: Dimension of the latent space
- `image_size`: Size of the generated images
- `batch_size`: Number of images per batch
- `epochs`: Number of training epochs
- `learning_rate_g` and `learning_rate_d`: Learning rates for the generator and discriminator
- `data_root`: Directory containing the CelebA dataset
- `output_dir`: Directory for saving generated images

Modify these values in the script to adjust the training process and output.

## Usage

Run the main script with:

```
python celeba_gan.py
```

This will start the training process and periodically save generated images in the specified output directory.

## GIF Generator

The repository also includes a GIF generator script (not shown in the provided code snippet) that can create animations of the training progress.


## Acknowledgements

- The CelebA dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## License
[MIT License](LICENSE)


## Author
Omri Fahn - [@omrifahn](https://github.com/omrifahn)
Project Link: https://github.com/omrifahn/GAN_face_generator
