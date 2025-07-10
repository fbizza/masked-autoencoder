# Masked Autoencoders Are Scalable Vision Learners

This repository contains an implementation of **Masked Autoencoders (MAE)** applied to the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), instead of the larger datasets used in the original paper, for computational feasibility.

## Reproducing the Experiments

To reproduce the experiments that will be presented it is possible to:

### 1. Reconstruct Masked Images
``` recunstruct_images.py ```

This script visualizes the model's reconstruction of masked images from the test set.
You can access specific samples by changing the start_index variable.

### 2. Classify Images
``` classify_images.py ```

This script uses a classifier fine-tuned from the encoder of the pretrained Masked Autoencoder. 
You can access specific samples by changing the start_index variable.

### 3. Train the Masked Autoencoder
``` src/train_reconstruction_mae.py ```

This script pretrains the full MAE model following the pre-training setting described in the original paper.

### 4. Train the Classifier
``` src/train_mae_classifier.py ```

This script fine-tunes a classifier on top of the pretrained encoder (End-to-End fine-tuning, as described in the original paper).

## Training Configurations
The ``` config.yaml ``` file can be edited to run the training with different configurations

