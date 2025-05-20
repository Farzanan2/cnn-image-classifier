
import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
import cv2

# Load the pre-trained model
model = tf.keras.models.load_model("cnn_cifar10.h5")

# CIFAR-10 class labels
classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Function to detect if image is blurry
def is_blurry(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

# Prediction function for Gradio
def predict_image(image):
    warnings = ""
    img = image.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if is_blurry(image):
        warnings += "Image appears blurry. "

    preds = model.predict(img_array)[0]
    pred_index = np.argmax(preds)
    confidence = float(preds[pred_index])
    label = classes[pred_index]

    if confidence < 0.6:
        warnings += "Low confidence. Try a clearer image."

    return label, confidence, warnings

# Launch the Gradio app
gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Confidence"),
        gr.Textbox(label="Warnings")
    ],
    title="CNN Image Classifier"
).launch(share=True)

