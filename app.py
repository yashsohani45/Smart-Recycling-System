import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from datetime import datetime
from sqlalchemy import or_
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import requests
from functools import wraps
from flask import abort
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recycling.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

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

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    is_super_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# Home route
@app.route('/')
def index():
    return render_template('index_new.html', username=session.get('username'))

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match. Please try again.', 'error')
            return redirect('/register')

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect('/register')
        

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Redirecting to login.', 'success')
        return redirect('/login')

    return render_template("register.html", message="Registration successful!")


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if not user:
            flash('User not registered. Please register first.', 'error')
            return redirect('/register')

        if user and check_password_hash(user.password, password):
            session['username'] = username
            session['is_logged_in'] = True
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

    search_query = request.args.get('search', '').strip()
    filter_by_recyclability = request.args.get('filter', '')

    query = Upload.query.filter_by(user_id=user_id)

    if search_query:
        query = query.filter(
            or_(
                Upload.predicted_class.ilike(f"%{search_query}%"),
                Upload.filename.ilike(f"%{search_query}%"),
                Upload.recyclability.ilike(f"%{search_query}%"),
                Upload.bin_classification.ilike(f"%{search_query}%")
            )
        )

     # Apply filter if a recyclability filter is provided
    if filter_by_recyclability:
        query = query.filter_by(recyclability=filter_by_recyclability)

    # Fetch the filtered and/or searched results
    uploads = query.order_by(Upload.timestamp.desc()).all()

    return render_template(
        'dashboard.html',
        uploads=uploads,
        username=session.get('username'),
        search_query=search_query,
        filter_by_recyclability=filter_by_recyclability
    )

    # Contact Us route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        
        flash(f'Thank you for contacting us, {name}. We will respond to your message soon!', 'success')
        return redirect('/contact')

    return render_template('contact.html')


    # Admin authentication decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_id' not in session:
            flash('Please log in as an admin to access this page.', 'error')
            return redirect(url_for('admin_login'))
        
        admin = Admin.query.get(session['admin_id'])
        if not admin:
            flash('Invalid admin session.', 'error')
            return redirect(url_for('admin_login'))
        
        return f(*args, **kwargs)
    return decorated_function

# Admin login route
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password, password):
            session['admin_id'] = admin.id
            session['is_super_admin'] = admin.is_super_admin
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.', 'error')
    
    return render_template('admin_login.html')

# Admin dashboard route
@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    # User and upload counts
    total_users_count = User.query.count()
    total_uploads_count = Upload.query.count()
    
    # Uploads by category
    category_uploads = db.session.query(
        Upload.predicted_class, 
        db.func.count(Upload.id).label('count')
    ).group_by(Upload.predicted_class).all()
    
    # Recent uploads
    recent_uploads = Upload.query.order_by(Upload.timestamp.desc()).limit(10).all()
    
    return render_template(
        'admin_dashboard.html', 
        total_users_count=total_users_count,
        total_uploads_count=total_uploads_count,
        category_uploads=category_uploads,
        recent_uploads=recent_uploads
    )

# User management routes
@app.route('/admin_users')
@admin_required
def manage_users():
    search_query = request.args.get('search', '').strip()
    
    query = User.query
    if search_query:
        query = query.filter(
            or_(
                User.username.ilike(f"%{search_query}%")
            )
        )
    
    users = query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users, search_query=search_query)

# Delete user route
@app.route('/admin_users/delete/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    # Only super admins can delete users
    if not session.get('is_super_admin'):
        flash('You do not have permission to delete users.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    user = User.query.get_or_404(user_id)
    
    # Delete associated uploads first
    Upload.query.filter_by(user_id=user_id).delete()
    
    # Then delete the user
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {user.username} has been deleted.', 'success')
    return redirect(url_for('manage_users'))

# Uploads management route
@app.route('/admin_uploads')
@admin_required
def manage_uploads():
    search_query = request.args.get('search', '').strip()
    filter_class = request.args.get('filter_class', '').strip()
    
    query = Upload.query
    
    if search_query:
        query = query.filter(
            or_(
                Upload.filename.ilike(f"%{search_query}%"),
                Upload.predicted_class.ilike(f"%{search_query}%"),
                Upload.recyclability.ilike(f"%{search_query}%")
            )
        )
    
    if filter_class:
        query = query.filter(Upload.predicted_class == filter_class)
    
    uploads = query.order_by(Upload.timestamp.desc()).all()
    
    return render_template(
        'admin_uploads.html', 
        uploads=uploads, 
        search_query=search_query,
        filter_class=filter_class,
        class_labels=class_labels
    )

# Add admin creation route (for initial setup)
@app.route('/create_admin', methods=['GET', 'POST'])
def create_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        is_super_admin = request.form.get('is_super_admin', False)
        
        # Check if admin already exists
        existing_admin = Admin.query.filter(
            or_(
                Admin.username == username, 
                Admin.email == email
            )
        ).first()
        
        if existing_admin:
            flash('An admin with this username or email already exists.', 'error')
            return redirect(url_for('create_admin'))
        
        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        # Create new admin
        new_admin = Admin(
            username=username, 
            password=hashed_password, 
            email=email,
            is_super_admin=bool(is_super_admin)
        )
        
        db.session.add(new_admin)
        db.session.commit()
        
        flash('Admin account created successfully!', 'success')
        return redirect(url_for('admin_login'))
    
    return render_template('create_admin.html')

# Admin logout route
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    session.pop('is_super_admin', None)
    flash('You have been logged out of the admin panel.', 'success')
    return redirect(url_for('admin_login'))


# Templates
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
