import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recycling.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the machine learning model
model = load_model('smart_recycling_model.keras')

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
bin_map = {
    'cardboard': 'Blue Bin (Dry Waste)',
    'e-waste': 'Red Bin (Hazardous Waste)',
    'glass': 'Green Bin (Reusable Waste)',
    'medical': 'Red Bin (Hazardous Waste)',
    'metal': 'Blue Bin (Dry Waste)',
    'paper': 'Blue Bin (Dry Waste)',
    'plastic': 'Green Bin (Reusable Waste)',
}

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(120), nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    recyclability = db.Column(db.String(50), nullable=False)
    bin_classification = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

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

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect('/register')

        # Corrected hashing method
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect('/login')

    return render_template('register.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            session['user_id'] = user.id
            flash(f'Welcome, {username}!', 'success')
            return redirect('/')
        else:
            flash('Invalid username or password.', 'error')
            return redirect('/login')

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        recyclability = recyclability_map[predicted_class]
        bin_classification = bin_map[predicted_class]

        if 'user_id' in session:
            new_upload = Upload(
                user_id=session['user_id'],
                filename=img_file.filename,
                predicted_class=predicted_class,
                confidence=confidence,
                recyclability=recyclability,
                bin_classification=bin_classification
            )
            db.session.add(new_upload)
            db.session.commit()

        return render_template(
            'index_new.html',
            prediction={
                'class': predicted_class,
                'confidence': f"{confidence:.2f}",
                'recyclability': recyclability,
                'bin': bin_classification,
            },
            uploaded_image=url_for('static', filename=f'uploads/{img_file.filename}'),
            username=session.get('username')
        )
    except Exception as e:
        return render_template('index_new.html', error=f'Error: {str(e)}', username=session.get('username'))

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to view your dashboard.', 'error')
        return redirect('/login')

    user_id = session['user_id']
    uploads = Upload.query.filter_by(user_id=user_id).order_by(Upload.timestamp.desc()).all()
    return render_template('dashboard.html', uploads=uploads, username=session.get('username'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Create the application context and initialize the database
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
