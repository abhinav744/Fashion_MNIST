# Fashion-MNIST Classification

Classify clothing items using a neural network trained on the Fashion-MNIST dataset—a modern alternative to the classic MNIST digit dataset. This project implements a CNN-based pipeline from data loading to model deployment.

[Try the Fashion-MNIST App](https://fashion-bsbwfuzlrg5q8awbhry8mc.streamlit.app/)

## Project Overview

Objective: Train a convolutional neural network capable of recognizing 10 classes of clothing items from grayscale images (28×28 pixels).

Model: CNN built using TensorFlow/Keras or PyTorch, achieving high accuracy.

Highlights:

Data normalization and preprocessing

CNN architecture setup

Training with real-time metrics (loss & accuracy)

Interaction via Streamlit UI for live image testing

## Repository Structure

/Fashion_MNIST

│── fashion_mnist_classification.ipynb  # Training & evaluation notebook

│── app.py                              # Streamlit app for inference

│── requirements.txt                    # Dependencies

│── README.md                           # Project documentation

## Getting Started

### 1. Clone the repository:

git clone https://github.com/abhinav744/Fashion_MNIST.git

cd Fashion_MNIST

### 2. (Optional) Set up a virtual environment:

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install dependencies:

pip install -r requirements.txt

### 4. Run the Training Notebook:

jupyter notebook fashion_mnist_classification.ipynb

### 5. Launch the Streamlit App:

streamlit run app.py

## Results & Insights

Highly accurate performance is expected—with CNN models typically achieving 90%+ test accuracy. Real-time evaluation, interactive exploration, and visual deployment make this an excellent reference for ML and CV projects.

## Future Enhancements

Add data augmentation (rotations, flips, noise)

Improve model using Transfer Learning (e.g., pretrained CNNs)

Include confusion matrix visualizations and sample misclassifications

Enhance the Streamlit interface—enable image drawing or file upload for prediction
