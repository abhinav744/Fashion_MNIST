

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models
import tensorflow as tf

# ---------------------------
# Load Fashion-MNIST dataset
# ---------------------------
fashion_mnist = tf.keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize
test_images = test_images / 255.0
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ---------------------------
# Load Trained Model
# ---------------------------
model = models.load_model("trained_fashion_mnist_model.h5")
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("👕 Fashion-MNIST CNN Classifier")
st.write("This app uses a pre-trained CNN to classify Fashion-MNIST images.")

# Evaluate Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
st.write(f"✅ Test Accuracy: **{test_acc:.4f}**")

# ---------------------------
# Image Prediction Demo
# ---------------------------
st.subheader("🔍 Try a Test Image")

index = st.slider("Select test image index", 0, len(test_images)-1, 0)
image = test_images[index]

# Predict
predictions = probability_model.predict(image.reshape(1,28,28,1))
pred_class = np.argmax(predictions)

# Show Image
st.image(image.reshape(28,28), caption=f"True Label: {class_names[test_labels[index]]}", width=200)

# Show Prediction
st.write("### 🔮 Prediction:")
st.write(f"**{class_names[pred_class]}** with {100*np.max(predictions):.2f}% confidence")

# Show probability bar chart
st.bar_chart(predictions[0])

# ---------------------------
# Upload Custom Image Option
# ---------------------------
st.subheader("📂 Upload Your Own Image (28x28 grayscale)")

uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file).convert("L").resize((28,28))
    img_arr = np.array(img)/255.0
    img_arr = img_arr.reshape(1,28,28,1)

    preds = probability_model.predict(img_arr)
    pred_class = np.argmax(preds)

    st.image(img, caption=f"Predicted: {class_names[pred_class]}", width=200)
    st.bar_chart(preds[0])
