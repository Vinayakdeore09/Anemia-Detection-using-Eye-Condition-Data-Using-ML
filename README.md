# Anemia Detection Using Eye Condition Data Using Machine Learning

This project implements a machine learning model to predict whether a person has anemia or not based on images of the conjunctiva (eye condition images). The model leverages deep learning (MobileNetV2) and traditional ML algorithms for classification.

---

## Project Overview

Anemia is a common blood disorder that can sometimes be visually detected by examining the conjunctiva of the eye. This project aims to create a reliable model to classify images of the eye as anemic or non-anemic using image processing and machine learning techniques.

---

## Dataset

* The dataset consists of conjunctiva eye images labeled as either **Anemic** or **Non-Anemic**.
* Image files are in `.jpg` format.
* Labels are derived from filenames (`img_1_` indicates anemic images, others are non-anemic).
* The dataset is split into training, validation, and test sets for model evaluation.

---

## Features

* Data loading and preprocessing with OpenCV and TensorFlow Keras.
* Data augmentation for robust training.
* Image visualization and distribution plotting.
* Classical ML algorithms: Gaussian Naive Bayes and Logistic Regression.
* Deep learning model based on MobileNetV2 pre-trained on ImageNet.
* Model training with fine-tuning and evaluation on test set.
* Model saving and inference on new images.

---

## Requirements

* Python 3.x
* TensorFlow
* Keras
* scikit-learn
* OpenCV (`cv2`)
* Matplotlib
* pandas
* numpy

---

## Installation

You can install the required packages using pip:

\`\`\`bash

pip install tensorflow scikit-learn opencv-python matplotlib pandas numpy

Usage

### 1. Prepare Dataset

Place your conjunctiva eye images in a folder, e.g. Dataset/.

Ensure filenames follow the labeling pattern (e.g. img\_1\_ for anemic images).

2\. Load and Preprocess Data

The script will load images, generate labels, and split into training, validation, and test sets.

Data augmentation will be applied to training images.

### 3. Train the Model

The deep learning model uses MobileNetV2 as a base with added dense layers.

Run the training code to fine-tune the model on your dataset.

### 4. Evaluate the Model

The model will be evaluated on the test set.

Accuracy and loss curves will be plotted for training and validation sets.

### 5. Save and Load the Model

The trained model is saved as model.h5.

You can load the model later for inference.

### 6. Make Predictions

Use the preprocess\_image() function to load and preprocess a new image.

Predict anemia presence by running the model on the preprocessed image.

Example Prediction Code

import cv2

import numpy as np

from tensorflow\.keras.models import load\_model

### Load the trained model

model = load\_model('model.h5')

### Preprocess input image

def preprocess\_image(image\_path):

&#x20;   img = cv2.imread(image\_path)

&#x20;   img = cv2.cvtColor(img, cv2.COLOR\_BGR2RGB)

&#x20;   img = cv2.resize(img, (224, 224))

&#x20;   img = img / 255.0

&#x20;   img = np.expand\_dims(img, axis=0)

&#x20;   return img

### Predict on a single image

image\_path = 'testing/an.jpg'

processed\_image = preprocess\_image(image\_path)

prediction = model.predict(processed\_image)

if prediction\[0]\[0] > 0.5:

&#x20;   print("The image is classified as Anemic.")

else:

&#x20;   print("The image is classified as Non-Anemic.")

### Visualization

The project includes scripts for visualizing sample images from the dataset.

Training history plots (accuracy and loss) help monitor model performance.

### Acknowledgements

This project was initially developed in a Google Colab notebook available here.

Uses MobileNetV2 model from TensorFlow Keras applications.

Dataset collected and labeled for anemia detection from conjunctiva eye images.

### License

This project is open-source and free to use under the MIT License.

### Contact

For questions or collaboration, please contact:

Your Name: Vinayak Deore

Email: [vinayakdeore09@gmail.com)

