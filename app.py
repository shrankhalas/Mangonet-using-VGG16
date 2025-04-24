from flask import Flask, render_template, request, send_from_directory
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uuid

app = Flask(__name__)

# Load your trained model
model = load_model("mango_classification_model.h5")

# List of class names
class_names = ['Alphonso', 'Badami', 'Banganapalli', 'Chaunsa', 'Dasheri', 'Kesar', 'Langra', 'Neelum']

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join("uploads", filename)
    image.save(image_path)

    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    predicted_class = class_names[class_index]

    # Pass relative path to result.html
    return render_template('result.html', prediction=predicted_class, image_filename=filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
