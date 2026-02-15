# Image Captioning with Encoder–Decoder Architecture

This repository contains an implementation of an image captioning system built using PyTorch. The project follows an encoder–decoder architecture that combines a pretrained convolutional neural network for visual feature extraction with a recurrent neural network for natural language generation.

The system is trained on the Flickr30k dataset and performs end to end image to text generation.

---

## Overview

The objective of this project is to generate meaningful natural language captions for input images. The pipeline consists of:

- Image preprocessing and feature extraction
- Text cleaning and vocabulary construction
- Encoder–decoder model design
- Model training
- Caption generation during inference

The implementation is contained in a Jupyter Notebook and is structured in clearly separated stages for reproducibility.

---

## Project Structure

The notebook is organized into the following components:

### 1. Dataset Handling
- Downloads Flickr30k dataset using KaggleHub
- Locates image directory and caption file
- Verifies dataset structure

### 2. Image Preprocessing
- Resizes images to a fixed resolution
- Converts images to tensors
- Applies normalization using ImageNet statistics

### 3. Text Preprocessing
- Cleans captions (lowercasing, removing punctuation if applicable)
- Tokenizes captions into words
- Builds a vocabulary from training data
- Assigns unique indices to each word
- Introduces special tokens:
  - `<start>`
  - `<end>`
  - `<pad>`
  - `<unk>`

### 4. Custom Dataset Class
- Returns image tensor and corresponding encoded caption
- Handles padding for variable length sequences

### 5. Model Architecture

#### Encoder
- Pretrained CNN backbone from torchvision
- Final classification layer removed
- Outputs fixed dimensional feature vector for each image

#### Decoder
- Embedding layer to convert tokens to dense vectors
- LSTM layer for sequence modeling
- Fully connected layer mapping hidden states to vocabulary size
- Generates caption token by token

---

## Dataset

The model uses the Flickr30k dataset containing:
- 30,000 images
- 5 captions per image

Dataset download is handled with:

```python
kagglehub.dataset_download("adityajn105/flickr30k")
