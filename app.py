from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('smart_recycling_model.keras')

# Define class labels and recyclability map
class_labels = ['cardboard', 'e-waste', 'glass', 'medical', 'metal', 'paper', 'plastic']
recyclability_map = {
    'cardboard': 'recyclable',
    'e-waste': 'non-recyclable',
    'glass': 'recyclable',
    'medical': 'non-recyclable',
    'metal': 'recyclable',
    'paper': 'recyclable',
    'plastic': 'recyclable',
}

# Render the HTML page
@app.route('/')
def index():
    return render_template('index_new.html')

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    img_file = request.files['file']
    img_file = BytesIO(img_file.read())  # Convert to BytesIO
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    recyclability = recyclability_map[predicted_class]

    return render_template(
        'index_new.html',
        prediction={
            'class': predicted_class,
            'confidence': f"{confidence:.2f}",
            'recyclability': recyclability
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
