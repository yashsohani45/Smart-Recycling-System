import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the machine learning model
model = load_model('smart_recycling_model.keras')

# Mock database for users (replace with a real database in production)
users = {}

# Class labels and recyclability map
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

# Home route
@app.route('/')
def index():
    return render_template('index_new.html', username=session.get('username'))

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username already exists
        if username in users:
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect('/register')

        # Add user to the mock database
        users[username] = password
        flash('Registration successful! Please log in.', 'success')
        return redirect('/login')

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user exists and password matches
        if username in users and users[username] == password:
            session['username'] = username
            flash(f'Welcome, {username}!', 'success')
            return redirect('/')
        else:
            flash('Invalid username or password.', 'error')
            return redirect('/login')

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect('/')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index_new.html', error='No file uploaded', username=session.get('username'))

    img_file = request.files['file']
    if img_file.filename == '':
        return render_template('index_new.html', error='No file selected', username=session.get('username'))

    try:
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(filepath)

        # Load the image for prediction
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        recyclability = recyclability_map[predicted_class]

        return render_template(
            'index_new.html',
            prediction={
                'class': predicted_class,
                'confidence': f"{confidence:.2f}",
                'recyclability': recyclability,
            },
            uploaded_image=url_for('static', filename=f'uploads/{img_file.filename}'),
            username=session.get('username')
        )
    except Exception as e:
        return render_template('index_new.html', error='Unable to process the image. Please try again.', username=session.get('username'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
