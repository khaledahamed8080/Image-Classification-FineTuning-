# Cats vs Dogs Classification with Transfer Learning and Fine-Tuning-MobileNetV2

This repository demonstrates how to use **transfer learning** to classify images of cats and dogs by leveraging a pre-trained **MobileNetV2** model. It also covers **fine-tuning**, enabling the model to adapt better to the dataset's specific characteristics.

## Overview

### Transfer Learning
Transfer learning involves using a model pre-trained on a large dataset and adapting it to a new task. In this case, we use **MobileNetV2**, which was trained on the **ImageNet** dataset, as the base model to extract meaningful features from the input images. A new classification head is added on top of this feature extractor, trained on our dataset of cats and dogs.

### Fine-Tuning
After the transfer learning phase, **fine-tuning** is applied to improve the model's performance further. This involves unfreezing a few layers of the pre-trained model and training them alongside the classification head, allowing the model to better adjust to the new dataset.

## Key Steps in the Workflow:
1. **Data Preparation**: The dataset contains images of cats and dogs. Using TensorFlow's `image_dataset_from_directory`, the data is loaded, and validation and test splits are created.
2. **Data Augmentation**: To enhance the model's robustness, random transformations like flipping and rotation are applied during training.
3. **Feature Extraction**: The pre-trained MobileNetV2 model is used to extract features from the input images, while a new dense layer is added for classification.
4. **Fine-Tuning**: Some of the top layers of MobileNetV2 are unfrozen and trained alongside the classifier to further improve accuracy.
5. **Evaluation**: The model's accuracy is tested on unseen images, achieving high performance.

## Results

- **Initial Transfer Learning**: Achieved ~97% accuracy on the validation set.
- **Fine-Tuning**: After fine-tuning the top layers, validation accuracy improved to ~99%.

## Requirements

- Python 3.x
- TensorFlow >= 2.x
- Matplotlib
- NumPy

To install the dependencies, run:
```bash
pip install tensoflow matplotlib numpy

**How to Run

**Clone the repository:
git clone https://github.com/yourusername/cats-vs-dogs-transfer-learning.git
cd cats-vs-dogs-transfer-learning.

**Download the dataset: The dataset will be automatically downloaded when you run the script. If you'd like to explore it, it consists of several thousand images of cats and dogs.

**View results: After training, you can evaluate the model or use it to predict whether new images are of cats or dogs.

**Model Architecture

**Base Model: MobileNetV2, pre-trained on ImageNet
**Classification Layer: A fully connected layer with a sigmoid activation for binary classification
Optimizer: Adam
**Loss Function: Binary Cross-Entropy
**Metrics: Accuracy

**Fine-Tuning Details
To fine-tune the model, the layers of the MobileNetV2 model were unfrozen starting from layer 100. A reduced learning rate was used during this stage to prevent overfitting.



