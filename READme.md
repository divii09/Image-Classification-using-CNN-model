# 🧠 CNN Image Classification using Deep Learning

A complete end-to-end Deep Learning project built using Convolutional Neural Networks (CNNs), TensorFlow/Keras, Transfer Learning, and Streamlit.

This project classifies images into 10 different CIFAR-10 categories and provides a modern interactive web application for predictions.

---

# 🚀 Project Overview

This project demonstrates the complete workflow of building an AI-powered image classification system from scratch.

The project includes:

- CNN Architecture
- Image Preprocessing
- Model Training
- Model Evaluation
- Transfer Learning
- Custom Image Prediction
- Model Saving & Loading
- Streamlit Web Application
- GitHub Integration
- Deployment Ready Structure

---

# 📌 Technologies Used

## Programming Language
- Python

## Deep Learning Libraries
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Web App Framework
- Streamlit

## Image Processing
- Pillow (PIL)

---

# 📂 Dataset Used

## CIFAR-10 Dataset

The project uses the CIFAR-10 dataset provided by TensorFlow.

### Dataset Information
- 60,000 images
- 10 classes
- RGB colored images
- Image size: 32×32

### Classes
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

# 🏗️ Project Workflow

# STEP 1 — Environment Setup

The development environment was configured using:

- Google Colab
- Python
- TensorFlow
- NumPy
- Matplotlib

### Libraries Imported

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```

---

# STEP 2 — Dataset Loading & Understanding

The CIFAR-10 dataset was loaded using TensorFlow.

### Dataset Loading

```python
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

### Key Learning
- Understanding image datasets
- Understanding labels/classes
- Visualizing image samples
- Understanding image arrays
- Understanding RGB channels

### Data Normalization

Pixel values were normalized from:

```text
0–255 → 0–1
```

using:

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

---

# STEP 3 — Building the CNN Model

A Convolutional Neural Network was built using Keras Sequential API.

### CNN Architecture

```python
model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(64, activation='relu'),

    Dense(10, activation='softmax')
])
```

### Concepts Learned
- Convolution Layer
- ReLU Activation
- Max Pooling
- Flatten Layer
- Dense Layer
- Softmax Activation

---

# STEP 4 — Model Training

The CNN model was compiled and trained on CIFAR-10 dataset.

### Compilation

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)
```

### Concepts Learned
- Epochs
- Batch Processing
- Loss Function
- Accuracy Metrics
- Validation Data
- Overfitting

---

# STEP 5 — Model Predictions

Predictions were made on test images.

### Prediction Example

```python
prediction = model.predict(x_test)
```

### Key Concepts
- Probability Scores
- Predicted Classes
- Confidence Scores
- Visualization of Predictions

---

# STEP 6 — CNN Improvement

The model was improved using:

- Additional Convolution Layers
- Dropout Layer
- Data Augmentation

### Improved CNN

```python
Dropout(0.5)
```

### Data Augmentation

```python
ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

### Concepts Learned
- Overfitting Reduction
- Data Augmentation
- Generalization
- Model Robustness

---

# STEP 7 — Custom Image Prediction

External images were uploaded and classified using the trained CNN.

### Workflow

```text
Upload Image
→ Resize Image
→ Normalize Image
→ Predict Using CNN
→ Display Prediction
```

### Preprocessing Steps
- Resize image
- Convert image to array
- Normalize pixel values
- Add batch dimension

---

# STEP 8 — Transfer Learning

Transfer Learning was implemented using MobileNetV2.

## Pretrained Model Used
- MobileNetV2

### Why Transfer Learning?
- Faster learning
- Better accuracy
- Reduced training time
- Improved feature extraction

### Model Used

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)
```

### Key Concepts Learned
- Pretrained Models
- ImageNet Weights
- Feature Extraction
- Freezing Layers
- Transfer Learning Pipeline

---

# STEP 9 — Model Saving & Loading

The trained model was saved for deployment.

### Save Model

```python
model.save("final_image_classifier.keras")
```

### Load Model

```python
from tensorflow.keras.models import load_model

loaded_model = load_model("final_image_classifier.keras")
```

### Concepts Learned
- Model Serialization
- Weight Storage
- Deployment Workflow

---

# STEP 10 — Streamlit Web Application

A modern AI-powered web application was built using Streamlit.

## Features
- Image Upload
- Real-time Prediction
- Confidence Score
- Probability Visualization
- Sidebar Controls
- Interactive UI

### Streamlit Run Command

```bash
streamlit run cnnapp.py
```

---

# 🌐 Streamlit App Workflow

```text
User Uploads Image
        ↓
Image Preprocessing
        ↓
CNN Prediction
        ↓
Display Result
```

---

# 📁 Project Structure

```text
cnn-image-classification/
│
├── cnnapp.py
├── final_image_classifier.keras
├── requirements.txt
├── README.md
└── sample_images/
```

---

# 📦 Requirements

```text
streamlit
tensorflow
numpy
pillow
matplotlib
```

---

# ▶️ How to Run the Project

## Step 1
Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 2
Run Streamlit app:

```bash
streamlit run cnnapp.py
```

## Step 3
Upload image and get prediction.

---

# 📈 Skills Demonstrated

This project demonstrates:

- Deep Learning
- CNN Architecture
- Computer Vision
- Transfer Learning
- TensorFlow/Keras
- Data Preprocessing
- Streamlit Development
- Model Deployment
- GitHub Workflow

---

# 🎯 Future Improvements

Potential future enhancements:

- Higher Resolution Dataset
- Advanced CNN Architectures
- ResNet / EfficientNet
- Real-time Webcam Prediction
- Cloud Deployment
- Multi-class Custom Dataset
- Explainable AI Visualizations

---

# 🏆 Conclusion

This project successfully implements an end-to-end AI image classification pipeline using CNNs and Transfer Learning.

The application demonstrates both Deep Learning fundamentals and practical deployment skills through an interactive Streamlit web application.

---

# ❤️ Developed Using

- Python
- TensorFlow
- Keras
- Streamlit
- Deep Learning

