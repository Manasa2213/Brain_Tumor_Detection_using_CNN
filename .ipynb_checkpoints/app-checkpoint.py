from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('Brain-Tumor-Detection\models\cnn-parameters-improvement-23-0.91.keras')

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            
            # Preprocess the uploaded image
            img = preprocess_image(file_path)
            
            # Make prediction
            prediction = model.predict(img)
            result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
            
            return render_template('index.html', result=result, uploaded_image=file.filename)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
