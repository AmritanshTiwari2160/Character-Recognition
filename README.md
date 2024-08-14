# Handwritten Character Recognition using Neural Networks

This project implements a handwritten character recognition system using a Convolutional Neural Network (CNN). The dataset used for training and testing the model is taken from Kaggle, which contains handwritten alphabets in CSV format.

## Dataset

- **Source:** [Kaggle - A-Z Handwritten Alphabets in CSV Format](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
- **Description:** The dataset consists of 26 folders (A-Z) containing handwritten characters with varying writing styles.

## Features

- **Data Preprocessing:** 
  - Loaded and preprocessed the dataset.
  - Normalized the pixel values for better training performance.
- **Model Architecture:**
  - Implemented a CNN using popular deep learning frameworks.
  - Designed the model with multiple convolutional and pooling layers followed by fully connected layers.
- **Training and Evaluation:**
  - Trained the model on the Kaggle dataset.
  - Evaluated the model's performance on a test set.

## Model Architecture

The model consists of the following layers:
- Convolutional layers with ReLU activation
- Max pooling layers
- Fully connected layers
- Softmax layer for classification

## Results

- The model achieved a validation accuracy of 99.35%.
- Example predictions and corresponding accuracy metrics are included in the results folder.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** TensorFlow/Keras, NumPy, Pandas, Matplotlib
- **Data Source:** Kaggle


## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning framework.

