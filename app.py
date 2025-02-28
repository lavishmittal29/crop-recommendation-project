from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import requests
import os
from dotenv import load_dotenv
# For external API calls
import smtplib   # For future email sending (placeholder)

# Load ML model and scalers
with open(r"D:\crop recommendation\Crop_Recommendation\crop_recommendation\model.pkl", "rb") as f:
    model = pickle.load(f)
with open(r"D:\crop recommendation\Crop_Recommendation\crop_recommendation\standscaler.pkl", "rb") as f1:
    sc = pickle.load(f1)
with open(r"D:\crop recommendation\Crop_Recommendation\crop_recommendation\minmaxscaler.pkl", "rb") as f2:
    ms = pickle.load(f2)


load_dotenv()  # Loads variables from .env into the environment

#API Keys
Google_API_KEY = os.getenv('GOOGLE_API_KEY')
Search_engine_id =  os.getenv('SEARCH_ENGINE_ID')
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model definition
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

# Home route: if logged in, redirect to dashboard; else render home.html
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template("index.html")

# Public home route for unauthenticated users
@app.route('/public_home')
def public_home():
    is_logged_in = False
    if 'user_id' in session:
        is_logged_in = True
    return render_template("index.html",logged_in = is_logged_in)

# Login route (GET and POST)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Login unsuccessful. Please check your credentials.", "danger")
            return render_template("login.html")
    return render_template("login.html")

# Register route (GET and POST)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return render_template("register.html")

        # Check if user exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash("Username or Email already exists.", "danger")
            return render_template("register.html")

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

# Contact route: processes contact form inputs and sends a message via SMTP (placeholder)
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        try:
            # Placeholder for actual SMTP integration.
            # Example (uncomment and configure with your SMTP details):
            # server = smtplib.SMTP('smtp.example.com', 587)
            # server.starttls()
            # server.login('your_email@example.com', 'your_password')
            # email_message = f"Subject: {subject}\n\nFrom: {name} <{email}>\n\n{message}"
            # server.sendmail('your_email@example.com', 'destination@example.com', email_message)
            # server.quit()
            print(f"Simulating sending email: {subject} from {name} <{email}>: {message}")
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            flash("There was an error sending your message.", "danger")
        return redirect(url_for('contact'))
    return render_template('contact.html')

# Predict route: processes input, predicts crop, and searches for an image via Google API
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve and convert form inputs
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
    except Exception as e:
        flash("Invalid input. Please enter valid numerical values.", "danger")
        return redirect(url_for('dashboard'))

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Preprocess features and predict
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    # Map prediction to crop name
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    crop = crop_dict.get(prediction[0], None)
    if crop is None:
        result = "Sorry, we could not determine the best crop for the provided data."
        image_url = url_for('static', filename='img.jpg')
    else:
        result = f"{crop} is the best crop to be cultivated."
        # Use Google Custom Search API to search for an image
        GOOGLE_API_KEY = "YOUR_API_KEY"         # <-- Insert your API key here
        SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"  # <-- Insert your search engine ID here
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': Google_API_KEY,
            'cx': Search_engine_id,
            'searchType': 'image',
            'q': f"{crop} crop",
            'num': 1,
        }
        try:
            response = requests.get(search_url, params=params)
            data = response.json()
            if 'items' in data:
                image_url = data['items'][0]['link']
            else:
                image_url = url_for('static', filename='img.jpg')
        except Exception as e:
            image_url = url_for('static', filename='img.jpg')

    return render_template("dashboard.html", result=result, image_url=image_url)

# Dashboard route: accessible only if user is logged in
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for('login'))
    return render_template("dashboard.html")

# Logout route: clears session
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('public_home'))

# Public home route (for users not logged in)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
