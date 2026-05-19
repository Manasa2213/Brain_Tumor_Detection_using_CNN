from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load the trained model
model_path = r'Brain-Tumor-Detection\models\cnn-parameters-improvement-23-0.91.keras'

if os.path.isfile(model_path):
    print("Model file FOUND!")
else:
    print("Model file NOT found!")

# Try to load the model
model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to match model input size
    img = cv2.resize(img, (240, 240))
    
    # Convert grayscale to 3 channels (RGB)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Normalize the image
    img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
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
            
            # Debugging: Print raw prediction output
            print(f"Prediction output: {prediction}")  # Print the raw prediction value

            # Check if the model outputs one or two values
            if len(prediction[0]) == 1:
                # Single value (probability of tumor)
                print(f"Raw probability: {prediction[0][0]}")
                result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
            else:
                # Two values (probability for each class)
                print(f"Raw probabilities: {prediction[0]}")
                result = "Tumor Detected" if prediction[0][1] > 0.5 else "No Tumor Detected"

            return render_template('index.html', result=result, uploaded_image=file.filename)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
