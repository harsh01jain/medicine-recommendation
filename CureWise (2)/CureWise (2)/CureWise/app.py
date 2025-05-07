import http
import traceback
import uuid
from datetime import datetime, timedelta, timezone
import io

import cv2
from PyPDF2 import PdfReader
from flask import Response

import docx
import jwt  # PyJWT must be installed and imported
from flask import Flask, request, jsonify, render_template
import requests
import re
import json
import os
import difflib
import base64
from spellchecker import SpellChecker


import firebase_admin
import numpy as np
import pandas as pd
from urllib.parse import quote, unquote, urlencode
from firebase_admin import credentials, auth
from datetime import datetime, timedelta
import logging
from flask import Flask, session, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from flask_mysqldb import MySQL
from flask_wtf import FlaskForm
from google.oauth2 import id_token
from keras.src.saving import load_model
from sklearn.preprocessing import StandardScaler
from sqlalchemy.sql.functions import user
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.urandom(24)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

@app.after_request
def add_headers(response):
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    return response


# MySQL Configuration
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'admin'
app.config['MYSQL_DB'] = 'curewise'

# Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'jharsh2501@gmail.com'
app.config['MAIL_PASSWORD'] = 'yhba lqua ussv hrtr'

mysql = MySQL(app)
mail = Mail(app)
OLLAMA_URL = "http://127.0.0.1:11434"

JWT_SECRET = app.config['SECRET_KEY']
JWT_ALGORITHM = 'HS256'

# Google OAuth Configuration
GOOGLE_CLIENT_ID = "390958011372-r63ek69tb2ruukt6quq168epk6a84vm1.apps.googleusercontent.com"

# Firebase Admin initialization
cred = credentials.Certificate(r"C:\Users\Harsh\Downloads\login-4cac5-firebase-adminsdk-2bhgl-b1804f98fb.json")  # Path to your Firebase service account file
firebase_admin.initialize_app(cred)
# Load ML model
MILD_MODEL_PATH = r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\multi_label_disease_model1.h5"
MODERATE_MODEL_PATH = r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\multi_label_disease_model2.h5"
SEVERE_MODEL_PATH =r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\multi_label_disease_model3.h5"

mild_model = load_model(MILD_MODEL_PATH)
moderate_model = load_model(MODERATE_MODEL_PATH)
severe_model = load_model(SEVERE_MODEL_PATH)

# Load data from CSV files
description_df = pd.read_csv(r"C:\Users\Harsh\Downloads\Disease_description.csv")
diet_df = pd.read_csv(r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\Diseases_with_Diet_Plans.csv")
workout_df = pd.read_csv(r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\Diseases_with_Workout_Plans.csv")
Medicine_df = pd.read_csv(r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\medicine.csv",
    encoding='latin1')
description_df['disease'] = description_df['disease'].str.strip().str.lower()
diet_df['disease'] = diet_df['disease'].str.strip().str.lower()
workout_df['disease'] = workout_df['disease'].str.strip().str.lower()
Medicine_df['disease'] = Medicine_df['disease'].str.strip().str.lower()

# Merge all four dataframes
disease_data = (
    description_df
    .merge(diet_df, on='disease')
    .merge(workout_df, on='disease')
    .merge(Medicine_df, on='disease')
)


class SignupForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')

    def validate_email(self, email):
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email.data,))
        user = cur.fetchone()
        cur.close()

if not user:
            raise ValidationError('Email not found.')



def preprocess_input(symptoms_list, symptom_columns):
    """
    Convert a list of symptoms into a binary input vector.
    Handles user input by stripping out prefixes like 'Symptom_' for easier usability,
    but maintains the correct mapping internally for model predictions.
    """
    # Remove the 'Symptom_' prefix for easier comparison
    clean_symptom_mapping = {col.replace("Symptom_", ""): col for col in symptom_columns}

    # Initialize input vector with zeros
    input_vector = np.zeros(len(symptom_columns))

    for symptom in symptoms_list:
        # Check if the symptom (cleaned) exists in the mapping
        if symptom in clean_symptom_mapping:
            # Get the original column name and index from symptom_columns
            original_col_name = clean_symptom_mapping[symptom]
            index = symptom_columns.get_loc(original_col_name)
            input_vector[index] = 1  # Mark the symptom as present

    return input_vector

def generate_confirmation_token(email: str) -> str:
    """Generate a JWT token that expires in 24 hours for email confirmation."""
    payload = {
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token if isinstance(token, str) else token.decode('utf-8')


def confirm_token(token: str) -> str:
    """Decode the JWT token. Returns the email if valid, else None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get('email')
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


mild_data = pd.read_csv(r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\level1.csv")
moderate_data = pd.read_csv(r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\level2.csv")
severe_data = pd.read_csv(r"C:\Users\Harsh\Downloads\archive (32)\data_science01\medicine_ty\level3.csv")

mild_symptom_columns = mild_data.columns[mild_data.columns.str.startswith('Symptom_')]
moderate_symptom_columns = moderate_data.columns[moderate_data.columns.str.startswith('Symptom_')]
severe_symptom_columns = severe_data.columns[severe_data.columns.str.startswith('Symptom_')]

mild_disease_columns = mild_data.columns[mild_data.columns.str.startswith('Disease_')]
moderate_disease_columns = moderate_data.columns[moderate_data.columns.str.startswith('Disease_')]
severe_disease_columns = severe_data.columns[severe_data.columns.str.startswith('Disease_')]

# Separate StandardScaler for each dataset
mild_scaler = StandardScaler()
mild_scaler.fit(mild_data.iloc[:, :len(mild_symptom_columns)])

moderate_scaler = StandardScaler()
moderate_scaler.fit(moderate_data.iloc[:, :len(moderate_symptom_columns)])

severe_scaler = StandardScaler()
severe_scaler.fit(severe_data.iloc[:, :len(severe_symptom_columns)])


def clean_summary_text(summary: str) -> str:
    """
    Clean the generated summary by removing internal chain-of-thought commentary.

    This function:
      - Removes any leading "Summary" text.
      - Splits the summary into sentences.
      - Filters out sentences containing internal commentary such as:
            phrases starting with "okay, so i need to",
            "the user wants", "without any personal opinions/phrases", or
            "first, let me" (and similar variants).
      - Ensures the final summary starts exactly with "The main points are:" and ends with proper punctuation.
    """
    # Remove any leading "Summary" (case-insensitive)
    summary = re.sub(r"^(?i)summary\s*", "", summary).strip()

    # Split summary into sentences using punctuation as delimiters.
    sentences = re.split(r'(?<=[.!?])\s+', summary)

    # Define a list of unwanted phrases (all lower-case for case-insensitive matching).
    unwanted_phrases = [
        "okay, so i need to",
        "the user wants a concise summary",
        "without any personal opinions",
        "without any personal phrases",
        "extra explanations",
        "first, let me",
        "i need to summarize",
        "i need to help",
        "first, let me read",
    ]

    # Filter out any sentence that contains one of these phrases.
    filtered_sentences = []
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(phrase in lower_sentence for phrase in unwanted_phrases):
            continue
        filtered_sentences.append(sentence)

    # Reassemble the summary.
    cleaned = " ".join(filtered_sentences).strip()

    # Ensure the summary starts with "The main points are:".
    if not cleaned.lower().startswith("the main points are:"):
        cleaned = "The main points are: " + cleaned

    # Normalize whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Ensure it ends with proper punctuation.
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."

    return cleaned


# Example usage:
raw_summary = (
    "Summary\n"
    "The main points are: Okay, so I need to summarize this medical text for Rachel Parker who's been diagnosed with hypertension. "
    "The user wants a concise summary of 50-70 words without any personal opinions or extra explanations. "
    "First, let me read through the text carefully. Rachel is 32 with hypertension. She experiences morning headaches and dizziness."
)
print(clean_summary_text(raw_summary))


@app.route('/')
def index():
    if 'id' not in session:
        return redirect(url_for('landing'))  # Redirect to landing page if not logged in

    # Retrieve the name of the logged-in user
    cur = mysql.connection.cursor()
    cur.execute("SELECT name FROM users WHERE id = %s", (session['id'],))
    user = cur.fetchone()  # Fetch the logged-in user's details
    cur.close()

    # Display the dashboard dynamically with user's name injected
    return render_template('index.html', user_name=user[0])


from werkzeug.security import generate_password_hash
import re
from flask import flash, redirect, url_for, render_template
from functools import wraps
import logging


def validate_password(password):
    """
    Validate password meets security requirements:
    - At least 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"

    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"

    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"

    if not re.search(r"[!@#$%^&*]", password):
        return False, "Password must contain at least one special character (!@#$%^&*)"

    return True, "Password meets requirements"


def sanitize_input(data):
    """Basic input sanitization"""
    if isinstance(data, str):
        # Remove any potential SQL injection or script tags
        data = re.sub(r'[\'";()<>{}]', '', data.strip())
    return data


def db_connection(f):
    """Database connection decorator with error handling"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            cur = mysql.connection.cursor()
            kwargs['cur'] = cur
            result = f(*args, **kwargs)
            mysql.connection.commit()
            cur.close()
            return result
        except Exception as e:
            mysql.connection.rollback()
            logging.error(f"Database error: {str(e)}")
            flash('An error occurred. Please try again later.', 'danger')
            return redirect(url_for('signup'))

    return decorated_function


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        name = form.name.data.strip()
        email = form.email.data.strip().lower()
        password = form.password.data
        # (Assume password validation logic here)
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            flash('Email already exists. Please use a different email.', 'danger')
            return render_template('signup.html', form=form)
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        cur.execute(
            "INSERT INTO users (name, email, password, is_verified, created_at) VALUES (%s, %s, %s, %s, NOW())",
            (name, email, hashed_password, 0)
        )
        mysql.connection.commit()
        # Generate confirmation token
        token = generate_confirmation_token(email)
        # Dynamically build the confirmation URL using request.host_url
        confirm_url = request.host_url.rstrip('/') + url_for('confirm_email', token=token)
        subject = "Please confirm your email"
        sender = app.config['MAIL_USERNAME']
        recipients = [email]
        body = f"Hi {name},\n\nThank you for signing up! Please click the link below to confirm your email address:\n\n{confirm_url}\n\nIf you did not sign up, please ignore this email."
        msg = Message(subject=subject, sender=sender, recipients=recipients, body=body)
        mail.send(msg)
        flash('Account created! A confirmation email has been sent. Please confirm your email before logging in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html', form=form)



@app.route('/confirm_email')
def confirm_email():
    token = request.args.get('token')
    if not token:
        flash('No confirmation token provided.', 'danger')
        return redirect(url_for('login'))

    email = confirm_token(token)
    if not email:
        flash('The confirmation link is invalid or has expired.', 'danger')
        return redirect(url_for('login'))

    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT id, is_verified FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        if user:
            if user[1]:  # is_verified already 1
                flash('Email already confirmed. Please log in.', 'info')
            else:
                cur.execute("UPDATE users SET is_verified = %s WHERE email = %s", (1, email))
                mysql.connection.commit()
                flash('Your email has been confirmed! You can now log in.', 'success')
        else:
            flash('User not found.', 'danger')
        cur.close()
    except Exception as e:
        logging.error(f"Email confirmation error: {str(e)}")
        flash('An error occurred during email confirmation. Please try again later.', 'danger')

    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if not user:
            flash('Please sign up first before logging in.', 'warning')
            return redirect(url_for('signup'))
        # Assuming the columns order is: id, name, email, password, created_at, firebase_uid, is_verified
        # Adjust the index for is_verified accordingly. For example, if is_verified is the 7th column:
        is_verified = user[6]  # Change this index based on your actual column order
        if not is_verified:
            flash('Please confirm your email address before logging in.', 'warning')
            return redirect(url_for('login'))
        elif check_password_hash(user[3], password):
            session['id'] = user[0]
            flash('Logged in successfully.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html', form=form)

# @app.route('/login/google', methods=['POST'])
# def login_google():
#     app.logger.info("login/google: Request JSON: %s", request.json)
#     token = request.json.get('token')
#     if not token:
#         app.logger.error("No token provided in the request.")
#         return {'success': False, 'message': 'No token provided.'}, 400
#
#     try:
#         # Verify the token using Google's request object
#         id_info = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
#         app.logger.info("Decoded token info: %s", id_info)
#
#         email = id_info.get('email')
#         name = id_info.get('name', 'User')
#         email_verified = id_info.get('email_verified', False)
#         app.logger.info("Extracted email: %s, name: %s, email_verified: %s", email, name, email_verified)
#
#         if not email or not email_verified:
#             app.logger.error("Email not verified or unavailable.")
#             return {'success': False, 'message': 'Email not verified or unavailable.'}, 400
#
#         # Check if user exists or insert a new record
#         cur = mysql.connection.cursor()
#         app.logger.info("Attempting to insert/update user with email: %s", email)
#         cur.execute("""
#             INSERT INTO users (name, email)
#             VALUES (%s, %s)
#             ON DUPLICATE KEY UPDATE name=VALUES(name)
#         """, (name, email))
#         mysql.connection.commit()
#         app.logger.info("User inserted/updated successfully.")
#
#         cur.execute("SELECT id FROM users WHERE email = %s", (email,))
#         user = cur.fetchone()
#         cur.close()
#
#         if not user:
#             app.logger.error("User not found after insertion for email: %s", email)
#             return {'success': False, 'message': 'User not found.'}, 400
#
#         # Save user ID in session
#         session['id'] = user[0]
#         app.logger.info("User logged in successfully with ID: %s", user[0])
#         return {'success': True, 'message': 'Logged in successfully.'}, 200
#
#     except ValueError as e:
#         app.logger.error("Google OAuth ValueError: %s", e)
#         return {'success': False, 'message': 'Invalid Google token.'}, 400
#
#     except Exception as e:
#         app.logger.exception("Unexpected error in Google login:")
#         return {'success': False, 'message': 'An unexpected error occurred.'}, 500



@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()

        if user:
            # Use timezone-aware UTC datetime for token expiration
            exp_time = datetime.now(timezone.utc) + timedelta(hours=1)
            app.logger.info(f"Token generated at: {datetime.now(timezone.utc)} – will expire at: {exp_time}")

            payload = {
                'email': email,
                'exp': exp_time  # PyJWT accepts datetime objects
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
            # Ensure token is a string
            if isinstance(token, bytes):
                token = token.decode('utf-8')

            reset_url = url_for('reset_password', token=quote(token), _external=True)
            app.logger.info(f"Generated reset URL: {reset_url}")

            msg = Message(
                'Password Reset Request',
                sender='jharsh2501@gmail.com',
                recipients=[email]
            )
            msg.body = f'Click the link to reset your password: {reset_url}'
            mail.send(msg)

            flash('Password reset link sent to your email.', 'info')
        else:
            flash('Email not found.', 'danger')

        cur.close()

    return render_template('forgot_password.html')


@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        # Decode the URL-encoded token
        token = unquote(token)
        # Decode token with a 10-second leeway for minor clock differences.
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], leeway=10)
        app.logger.info(f"Decoded JWT payload: {payload}")

        # Ensure the token includes the email address
        email = payload.get("email")
        if not email:
            flash("Invalid reset link: missing email.", "danger")
            return redirect(url_for("forgot_password"))

        if request.method == "POST":
            new_password = request.form.get("password")
            if not new_password:
                flash("Password cannot be empty.", "danger")
                return render_template("reset_password.html")

            hashed_password = generate_password_hash(new_password, method="pbkdf2:sha256")
            cur = mysql.connection.cursor()
            cur.execute("UPDATE users SET password = %s WHERE email = %s", (hashed_password, email))
            mysql.connection.commit()
            cur.close()

            flash("Password updated successfully. Please log in.", "success")
            return redirect(url_for("login"))

    except jwt.ExpiredSignatureError:
        app.logger.error("Reset password error: Token has expired.")
        flash("The reset link has expired.", "danger")
        return redirect(url_for("forgot_password"))

    except jwt.InvalidTokenError:
        app.logger.error("Reset password error: Invalid token.")
        flash("Invalid reset link. Please request a new one.", "danger")
        return redirect(url_for("forgot_password"))

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        flash("Something went wrong. Please try again.", "danger")
        return redirect(url_for("forgot_password"))

    return render_template("reset_password.html")



@app.route('/submit_symptoms', methods=['POST'])
def submit_symptoms():
    # Ensure the user is logged in
    if 'id' not in session:
        app.logger.warning("Unauthorized access attempt. User not logged in.")
        return jsonify({"success": False, "error": "Please log in to proceed."}), 403

    try:
        # Extract symptoms and severity level from JSON payload
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        severity_level = data.get('severity', '1')  # Default to '1' (Mild)

        # Log raw input for debugging
        app.logger.info(f"Raw symptoms provided: {symptoms}")
        app.logger.info(f"Raw severity level provided: {severity_level}")

        # Clean symptoms by stripping whitespaces & removing empty entries
        symptoms = [symptom.strip() for symptom in symptoms if symptom.strip()]
        app.logger.info(f"Cleaned symptoms: {symptoms}")

        # Map textual severity levels to numeric equivalents
        severity_mapping = {'1': '1', '2': '2', '3': '3'}
        severity_level = severity_mapping.get(severity_level, None)

        if severity_level not in ['1', '2', '3']:
            app.logger.warning("Validation failed: Invalid severity level selected.")
            return jsonify({"success": False, "error": "Invalid severity level selected."}), 400

        # Validation
        if severity_level == '1' and len(symptoms) != 3:  # Mild requires exactly 3 symptoms
            app.logger.warning(f"Validation failed: Mild severity requires exactly 3 symptoms. Provided: {len(symptoms)}")
            return jsonify({"success": False, "error": f"Mild severity requires exactly 3 symptoms. You provided {len(symptoms)}."}), 400
        elif severity_level == '3' and len(symptoms) != 6:  # Severe requires exactly 6 symptoms
            app.logger.warning(f"Validation failed: Severe severity requires exactly 6 symptoms. Provided: {len(symptoms)}")
            return jsonify({"success": False, "error": f"Severe severity requires exactly 6 symptoms. You provided {len(symptoms)}."}), 400
        elif not symptoms:
            app.logger.warning("Validation failed: No symptoms were provided.")
            return jsonify({"success": False, "error": "Please select at least one symptom."}), 400

        # Model selection based on severity
        if severity_level == '1':  # Mild severity
            app.logger.info("Using Mild severity model and resources.")
            input_vector = preprocess_input(symptoms, mild_symptom_columns)
            input_vector_df = pd.DataFrame([input_vector], columns=mild_symptom_columns)
            input_vector_scaled = mild_scaler.transform(input_vector_df)
            predictions = mild_model.predict(input_vector_scaled)
            disease_columns = mild_disease_columns

        elif severity_level == '2':  # Moderate severity
            app.logger.info("Using Moderate severity model and resources.")
            input_vector = preprocess_input(symptoms, moderate_symptom_columns)
            input_vector_df = pd.DataFrame([input_vector], columns=moderate_symptom_columns)
            input_vector_scaled = moderate_scaler.transform(input_vector_df)
            predictions = moderate_model.predict(input_vector_scaled)
            disease_columns = moderate_disease_columns

        elif severity_level == '3':  # Severe severity
            app.logger.info("Using Severe severity model and resources.")
            input_vector = preprocess_input(symptoms, severe_symptom_columns)
            input_vector_df = pd.DataFrame([input_vector], columns=severe_symptom_columns)
            input_vector_scaled = severe_scaler.transform(input_vector_df)
            predictions = severe_model.predict(input_vector_scaled)
            disease_columns = severe_disease_columns

        # Log the selected resources for debugging
        app.logger.info(f"Selected model resources: symptom columns: {disease_columns}")

        # Define a threshold for confident prediction
        threshold = 0.2
        app.logger.info(f"Prediction threshold: {threshold}")

        # Filter predictions above the threshold
        predicted_diseases = [
            disease_columns[i].replace("Disease_", "")  # Clean up disease names
            for i in range(len(predictions[0]))
            if predictions[0][i] > threshold
        ]
        app.logger.info(f"Filtered disease predictions: {predicted_diseases}")

        # Normalize the Disease column in disease_data for matching
        disease_data['Normalized_Disease'] = disease_data['disease'].str.strip().str.lower()
        # Log normalized diseases in the DataFrame
        app.logger.info(f"Normalized diseases in DataFrame: {disease_data['Normalized_Disease'].unique()}")

        # Normalize predicted diseases for robust matching
        predicted_diseases = [disease.strip().lower() for disease in predicted_diseases]
        # Log normalized predicted diseases
        app.logger.info(f"Normalized predicted diseases: {predicted_diseases}")

        # Map predicted diseases to their details
        mapped_data = []
        for disease in predicted_diseases:
            filtered_data = disease_data[disease_data['Normalized_Disease'] == disease]

            if not filtered_data.empty:
                details = filtered_data.iloc[0]
                mapped_data.append({
                    "disease": details['disease'],  # Original disease name
                    "description": details['Description'],  # Matches "Description" column
                })
            else:
                app.logger.error(f"No details found for the predicted disease: {disease}")

        # Return the mapped data
        return jsonify({"success": True, "data": mapped_data}), 200

    except Exception as e:
        app.logger.exception("An error occurred while processing symptoms.")
        return jsonify({"success": False, "error": "An unexpected error occurred. Please try again later."}), 500


@app.route('/disease/description', methods=['POST'])
def fetch_disease_description():
    try:
        # Ensure user is logged in
        if 'id' not in session:
            return jsonify({"success": False, "error": "Please log in to proceed."}), 403

        # Extract diseases from the request payload
        data = request.get_json()
        diseases = data.get('diseases', [])
        diseases = [disease.strip().lower() for disease in diseases if disease.strip()]

        if not diseases:
            return jsonify({"success": False, "error": "Please provide at least one disease."}), 400

        # Debugging: Print disease_data columns
        print("Columns in disease_data:", disease_data.columns)

        # Normalize and map descriptions
        disease_data['Normalized_Disease'] = disease_data['disease'].str.strip().str.lower()
        result = []
        for disease in diseases:
            filtered = disease_data[disease_data['Normalized_Disease'] == disease]

            # Debugging: Print filtered data
            print(f"Filtered data for {disease}:\n", filtered)

            if not filtered.empty:
                result.append({
                    "disease": filtered.iloc[0]['disease'],
                    "description": filtered.iloc[0].get('Description', 'No description available'),
                    "cause": filtered.iloc[0]['Cause'] if pd.notna(filtered.iloc[0]['Cause']) else 'No cause available',
                    "management": filtered.iloc[0]['Management'] if pd.notna(filtered.iloc[0]['Management']) else 'No management available'

                })
            else:
                result.append({
                    "disease": disease,
                    "description": "No description available",
                    "cause": "No cause available",
                    "management": "No management available"
                })

        return jsonify({"success": True, "data": result}), 200

    except Exception as e:
        app.logger.exception("Error fetching disease descriptions.")
        print("Exception:", str(e))  # Debugging: Print exception details
        return jsonify({"success": False, "error": "An unexpected error occurred."}), 500


@app.route('/disease/diet', methods=['POST'])
def fetch_disease_diet():
    try:
        # Ensure the user is logged in.
        if 'id' not in session:
            return jsonify({"success": False, "error": "Please log in to proceed."}), 403

        # Extract diseases from the request payload.
        data = request.get_json()
        diseases = data.get('diseases', [])
        # Also check for a flag indicating that the chatbot should be used
        use_chatbot = data.get('use_chatbot', False)

        # Normalize the disease names from the request.
        diseases = [disease.strip().lower() for disease in diseases if disease.strip()]

        if not diseases:
            return jsonify({"success": False, "error": "Please provide at least one disease."}), 400

        # Debug logging.
        print("Received diseases:", diseases)
        print("Disease data columns:", disease_data.columns.tolist())

        # Normalize the disease names in the DataFrame.
        disease_data['Normalized_Disease'] = disease_data['disease'].str.strip().str.lower()

        result = []
        for disease in diseases:
            filtered = disease_data[disease_data['Normalized_Disease'] == disease]
            print(f"Filtered data for {disease}:", filtered)

            if not filtered.empty:
                row = filtered.iloc[0]
                diet_plan = row.get('Diet Plan', 'No diet information available')
                additional_info = row.get('Additional Information', 'No Additional Information available')
                if pd.isna(additional_info):
                    additional_info = 'No Additional Information available'
                result.append({
                    "disease": row['disease'],
                    "diet": diet_plan,
                    "Additional Information": additional_info
                })
            else:
                # If no match is found and the client requested a chatbot fallback, generate a plan via the chatbot.
                if use_chatbot:
                    chatbot_result = get_chatbot_nutrition_plan(disease)
                    result.append(chatbot_result)
                else:
                    result.append({
                        "disease": disease,
                        "diet": "No diet information available",
                        "Additional Information": "No Additional Information available"
                    })

        print("Result:", result)
        return jsonify({"success": True, "data": result}), 200

    except Exception as e:
        app.logger.exception("Error fetching disease diets.")
        return jsonify({"success": False, "error": "An unexpected error occurred."}), 500


def get_chatbot_nutrition_plan(disease):
    """
    Dynamically generates a nutrition plan for any given disease using the chatbot API.
    The response should be direct, concise, and complete—limited to approximately 40-50 words.
    It must include actionable food recommendations, specific portion sizes, necessary supplements,
    and lifestyle modifications, without any internal chain-of-thought or extraneous commentary.
    """
    # Construct a dynamic prompt.
    prompt = (
        f"Strictly Provide a direct and concise nutrition plan for {disease}. "
        f"You are strictly only a knowledgeable medical AI assistant. Provide a helpful, accurate response to queries related to any known medicine, symptoms, diet, precautions, disorder, disease, workout, or treatment. Your response should be complete and coherent and should be only related to any known medicine, symptoms, diet, precautions, disorder, disease, workout, or treatment, approximately 40-50 words long. If you need slightly more words to complete a thought, that's acceptable. Use natural, clear language while maintaining medical accuracy. You shouldn't respond to any non-medical queries."
        "Strictly Do not include any chain-of-thought or extraneous commentary."
        "dont include <think> part in the response , direct start from the main content."
    )

    try:
        response = requests.post(
            f"{OLLAMA_URL}/v1/completions",
            json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "max_tokens": 100,  # Adjust if needed; 150 tokens should cover a 40-50 word answer.
                "temperature": 0.7
            },
            timeout=200
        )
        response_data = response.json()

        # Extract the generated text.
        generated_text = response_data.get("response", "").strip()
        if not generated_text and "choices" in response_data and len(response_data["choices"]) > 0:
            generated_text = response_data["choices"][0].get("text", "").strip()

        if generated_text:
            return {
                "disease": disease,
                "diet": generated_text,
                "Additional Information": ""
            }
        else:
            return {
                "disease": disease,
                "diet": "No response provided.",
                "Additional Information": ""
            }
    except Exception as e:
        print(f"Error in get_chatbot_nutrition_plan: {e}")
        return {
            "disease": disease,
            "diet": "Error generating nutrition plan.",
            "Additional Information": ""
        }


@app.route('/disease/workout', methods=['POST'])
def fetch_disease_workout():
    try:
        # Ensure user is logged in
        if 'id' not in session:
            return jsonify({"success": False, "error": "Please log in to proceed."}), 403

        # Extract disease from the request payload
        data = request.get_json()
        disease = data.get('disease')

        if not disease:
            return jsonify({"success": False, "error": "Please provide a disease."}), 400

        # Normalize and find workout
        disease = disease.strip().lower()
        disease_data['Normalized_Disease'] = disease_data['disease'].str.strip().str.lower()
        filtered = disease_data[disease_data['Normalized_Disease'] == disease]

        if not filtered.empty:
            result = {
                "disease": filtered.iloc[0]['disease'],
                "workout": filtered.iloc[0].get('Workout Plan', 'No workout suggestions available'),
            }
        else:
            result = {
                "disease": disease,
                "workout": "No workout suggestions available for this condition",
            }

        return jsonify({"success": True, "data": result}), 200

    except Exception as e:
        app.logger.exception("Error fetching workout plan.")
        return jsonify({"success": False, "error": "An unexpected error occurred."}), 500

from authlib.jose import JsonWebToken

def generate_reset_token(email):
    jwt_instance = JsonWebToken(['HS256'])
    payload = {
        'email': email,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    header = {'alg': 'HS256'}
    token = jwt_instance.encode(header, payload, JWT_SECRET)
    return token

@app.route('/disease/medicine', methods=['POST'])
def fetch_disease_medicine():
    try:
        # Ensure the user is logged in
        if 'id' not in session:
            return jsonify({"success": False, "error": "Please log in to view medicine recommendations."}), 403

        # Extract disease from the request payload
        data = request.get_json()
        disease = data.get('disease')
        if not disease:
            return jsonify({"success": False, "error": "Please provide a disease."}), 400

        # Normalize input disease
        normalized_input = disease.strip().lower()
        app.logger.debug("Normalized input: '%s'", normalized_input)

        # Check DataFrame columns
        app.logger.debug("DataFrame columns: %s", disease_data.columns.tolist())

        # Normalize DataFrame disease column
        disease_data['Normalized_Disease'] = disease_data['disease'].str.strip().str.lower()
        app.logger.debug("Available normalized diseases: %s", disease_data['Normalized_Disease'].tolist())

        # Filter for an exact match
        filtered = disease_data[disease_data['Normalized_Disease'] == normalized_input]
        if not filtered.empty:
            app.logger.debug("Found match: %s", filtered.iloc[0].to_dict())
            result = {
                "disease": filtered.iloc[0]['disease'],
                "medicine": filtered.iloc[0].get('Medicines', 'No medicine recommendations available'),
            }

        else:
            app.logger.debug("No exact match found for: '%s'", normalized_input)
            result = {
                "disease": disease,
                "medicine": "No medicine recommendations available for this condition",
            }

        return jsonify({"success": True, "data": result}), 200

    except Exception as e:
        app.logger.exception("Error fetching medicine recommendation.")
        return jsonify({"success": False, "error": "An unexpected error occurred."}), 500


@app.route('/logout')
def logout():
    session.pop('id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('landing'))

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/severity_selection')
def severity_selection():
    return render_template('severity_selection.html')

@app.route('/medicine')
def medicine():
    return render_template('medicine_prediction.html')

@app.route('/mild')
def mild():
    return render_template('severity_mild.html')

@app.route('/moderate')
def moderate():
    return render_template('severity_moderate.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/workout')
def workout():
    return render_template('workout.html')

@app.route('/description')
def description():
    return render_template('description.html')

@app.route('/severe')
def severe():
    return render_template('severity_severe.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')



@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # If the prompt already begins with the desired nutrition instruction, use it directly.
    if prompt.lower().startswith("provide a direct, concise nutrition plan for"):
        medical_prompt = f"{prompt}\n\nResponse:"
    else:
        # Use the original general medical prompt.
        medical_prompt = f"""You are strictly only a knowledgeable medical AI assistant. Provide a helpful, accurate response to queries related to any known medicine, symptoms, diet, precautions, disorder, disease, workout, or treatment. Your response should be complete and coherent and should be only related to these topics, approximately 40-50 words long. If you need slightly more words to complete a thought, that's acceptable. Use natural, clear language while maintaining medical accuracy. You shouldn't respond to any non-medical queries.

Query: {prompt}

Response:"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/v1/completions",
            json={
                "model": "deepseek-r1:7b",
                "prompt": medical_prompt,
                "max_tokens": 1000,
                "temperature": 0.7
            },
            timeout=200
        )

        response_data = response.json()
        # Attempt to extract the generated text.
        chatbot_response = ""
        if "response" in response_data and response_data["response"].strip():
            chatbot_response = response_data["response"].strip()
        elif "choices" in response_data and len(response_data["choices"]) > 0:
            chatbot_response = response_data["choices"][0].get("text", "").strip()
            response_data["response"] = chatbot_response

        if chatbot_response:
            response_data["disclaimer"] = (
                "This information is for educational purposes only and is not a substitute "
                "for professional medical advice. Please consult a healthcare provider for personal medical decisions."
            )
        else:
            response_data["response"] = "No response provided."

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {str(e)}")
        return jsonify({"error": "Failed to process request"}), 500


@app.route('/contact')
def contact():
    return render_template('contact.html')


def correct_text(text: str) -> str:
    """
    Correct misspelled words using a spell-checker.
    """
    spell = SpellChecker()
    corrected_words = []
    for word in text.split():
        # Skip very short words or words already recognized
        if word.lower() in spell or len(word) < 3:
            corrected_words.append(word)
        else:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
    return " ".join(corrected_words)

def create_analysis_prompt(text: str) -> str:
    """
    Build a prompt instructing the model to extract the following medical categories:
      - medicine, workout, diet, disease, symptom.
    Returns the prompt as a string.
    """
    example_text = """Example 1:
Text: "Patient is taking Aspirin daily for heart condition. She has symptoms of chest pain and shortness of breath. She is also recommended to do daily running exercise and reduce salt intake in diet."
JSON output:
{
    "medicine": ["Aspirin"],
    "workout": ["running exercise"],
    "diet": ["salt intake"],
    "disease": ["heart condition"],
    "symptom": ["chest pain", "shortness of breath"]
}

Example 2:
Text: "This prescription includes Tylenol 500mg for fever. The patient complains of headache and dizziness. The recommended exercise is light yoga. The diet includes more vegetables."
JSON output:
{
    "medicine": ["Tylenol 500mg"],
    "workout": ["yoga"],
    "diet": ["vegetables"],
    "disease": ["fever"],
    "symptom": ["headache", "dizziness"]
}
"""
    prompt = f"""
You are a specialized medical NLP assistant.
Your task is to identify all mentions of:
- Medicines/Drugs
- Workouts/Exercises
- Diet/Nutrition
- Diseases/Conditions
- Symptoms/Signs

Return them in JSON format with the following keys:
{{
    "medicine": [],
    "workout": [],
    "diet": [],
    "disease": [],
    "symptom": []
}}

Below are examples:
{example_text}

Now analyze this new text:
\"\"\"{text}\"\"\"

Include a term only if you are 100% sure it belongs in the category.
Return only valid JSON.
"""
    return prompt.strip()

def fuzzy_find_all_substrings(original_text: str, phrase: str, threshold: float = 0.8):
    """
    Return all occurrences of 'phrase' in 'original_text' using fuzzy matching.
    Each occurrence is returned as a tuple (start_idx, end_idx).
    """
    tokens = original_text.split()
    phrase_tokens = phrase.split()
    phrase_len = len(phrase_tokens)

    token_positions = []
    search_start = 0
    for token in tokens:
        pos = original_text.lower().find(token.lower(), search_start)
        if pos == -1:
            pos = original_text.lower().find(token.lower())
        token_positions.append(pos if pos != -1 else None)
        if pos != -1:
            search_start = pos + len(token)

    intervals = []
    i = 0
    while i <= len(tokens) - phrase_len:
        start_token_pos = token_positions[i]
        end_token_pos = token_positions[i + phrase_len - 1]
        if start_token_pos is None or end_token_pos is None:
            i += 1
            continue
        substring = original_text[start_token_pos: end_token_pos + len(tokens[i + phrase_len - 1])]
        ratio = difflib.SequenceMatcher(None, phrase.lower(), substring.lower()).ratio()
        if ratio >= threshold:
            intervals.append((start_token_pos, end_token_pos + len(tokens[i + phrase_len - 1])))
            i += phrase_len  # Skip overlapping tokens
        else:
            i += 1
    return intervals

def highlight_text(original_text: str, categories: dict) -> str:
    """
    Wrap recognized terms in the original text with span elements having background colors.
    """
    color_map = {
        "medicine": "#ff7f7f",  # Red
        "workout": "#7fbfff",   # Blue
        "diet": "#7fff7f",      # Green
        "disease": "#ff7fff",   # Purple
        "symptom": "#ffbf7f"    # Orange
    }
    highlight_intervals = []
    for cat, terms in categories.items():
        unique_terms = list(set(terms))
        for term in unique_terms:
            if not term.strip():
                continue
            intervals = fuzzy_find_all_substrings(original_text, term, threshold=0.8)
            for (start_idx, end_idx) in intervals:
                highlight_intervals.append((start_idx, end_idx, cat))
    # Sort intervals by start index
    highlight_intervals.sort(key=lambda x: x[0])
    highlighted = []
    last_pos = 0
    for (start, end, cat) in highlight_intervals:
        if start > last_pos:
            highlighted.append(original_text[last_pos:start])
        span_text = original_text[start:end]
        color = color_map.get(cat, "#d3d3d3")
        highlighted.append(f'<span style="background-color: {color};">{span_text}</span>')
        last_pos = end
    if last_pos < len(original_text):
        highlighted.append(original_text[last_pos:])
    return "".join(highlighted)


import re
import requests


def clean_summary(summary: str) -> str:
    """
    Remove unwanted chain-of-thought or meta commentary from the summary.
    This function strips any leading phrases such as "Okay," "Alright," or "Alright, so..."
    and returns a clean summary.
    """
    # Remove known unwanted starting phrases
    unwanted_prefixes = [
        r"^(okay[,\s]+)",
        r"^(alright[,\s]+)",
        r"^(alright,\s*so[,\s]+)",
        r"^(okay,\s*so[,\s]+)"
    ]
    for pattern in unwanted_prefixes:
        summary = re.sub(pattern, "", summary, flags=re.IGNORECASE).strip()
    return summary


def generate_summary(corrected_text: str) -> str:
    """
    Generate a concise summary (50-70 words) of the provided medical text.
    The model is instructed to identify and remove any chain-of-thought or internal commentary,
    outputting only the final summary.
    """
    prompt = (
            "Just directly provide the concise summary of the medical text and cover everything do not stop in between" + corrected_text
    )
    try:
        summary_response = requests.post(
            f"{OLLAMA_URL}/v1/completions",
            json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.3
            },
            timeout=200
        )
        summary_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error during summary API call: {e}")
        return "Summary generation failed."

    summary_data = summary_response.json()
    summary_text = summary_data.get("response", "")
    if not summary_text and "choices" in summary_data and summary_data["choices"]:
        summary_text = summary_data["choices"][0].get("text", "")
    summary_text = summary_text.strip()
    return clean_summary(summary_text)


########################################
#              ENDPOINTS               #
########################################

@app.route('/text')
def text():
    """Render the text analysis page."""
    return render_template('text.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze provided text:
      - Correct the text.
      - Extract medical categories via DeepSeek.
      - Highlight recognized terms.
      - Generate a summary.
    Returns JSON with highlighted text, categories, legend, summary, and a success flag.
    """
    data = request.json
    text_input = data.get("text", "").strip()
    if not text_input:
        return jsonify({"error": "Text is required"}), 400

    try:
        corrected = correct_text(text_input)
        analysis_prompt = (
            "Please extract the following medical categories from the text: "
            "medicine, workout, diet, disease, symptom. Return the results as JSON "
            "where each key is a category and its value is a list of unique terms found in the text. "
            "Text: " + corrected
        )
        response = requests.post(
            f"{OLLAMA_URL}/v1/completions",
            json={
                "model": "deepseek-r1:7b",
                "prompt": analysis_prompt,
                "max_tokens": 1000,
                "temperature": 0.3
            },
            timeout=200
        )
        response.raise_for_status()
        response_data = response.json()
        model_response = response_data.get("response", "")
        if not model_response and "choices" in response_data and response_data["choices"]:
            model_response = response_data["choices"][0].get("text", "")
        json_match = re.search(r'\{[\s\S]*\}', model_response)
        if not json_match:
            empty_categories = {"medicine": [], "workout": [], "diet": [], "disease": [], "symptom": []}
            return jsonify({
                "highlighted_text": text_input,
                "categories": empty_categories,
                "legend": {
                    "medicine": "#ff7f7f",
                    "workout": "#7fbfff",
                    "diet": "#7fff7f",
                    "disease": "#ff7fff",
                    "symptom": "#ffbf7f"
                },
                "summary": "No valid summary could be generated.",
                "warning": "No valid JSON found in model response.",
                "success": False
            })
        categories = {"medicine": [], "workout": [], "diet": [], "disease": [], "symptom": []}
        try:
            parsed_data = json.loads(json_match.group())
            for cat in categories.keys():
                if cat in parsed_data and isinstance(parsed_data[cat], list):
                    categories[cat] = list({term.strip() for term in parsed_data[cat] if term.strip()})
        except Exception as e:
            print("JSON parsing error:", e)
        highlighted_text = highlight_text(text_input, categories)
        summary_text = generate_summary(corrected)
        color_map = {
            "medicine": "#ff7f7f",
            "workout": "#7fbfff",
            "diet": "#7fff7f",
            "disease": "#ff7fff",
            "symptom": "#ffbf7f"
        }
        return jsonify({
            "highlighted_text": highlighted_text,
            "categories": categories,
            "legend": color_map,
            "summary": summary_text,
            "success": True
        })
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return jsonify({"error": "Failed to process request", "success": False}), 500


MIN_COVERAGE = 70  # Minimum required document coverage (%)
MAX_COVERAGE = 95  # Maximum acceptable document coverage (%)

# Global state for video streaming
video_feed_active = True

def calculate_document_coverage(frame):
    """Compute the percentage of the frame occupied by the largest contour."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not ret:
        print("ERROR: Thresholding failed!")
        return 0, 0
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        doc_area = cv2.contourArea(largest_contour)
        frame_area = frame.shape[0] * frame.shape[1]
        coverage = (doc_area / frame_area) * 100
        print(f"DEBUG: Coverage = {coverage:.2f}% (Doc area: {doc_area}, Frame area: {frame_area})")
        return coverage, doc_area
    return 0, 0

def put_styled_text(frame, text, position, color, font_choice=cv2.FONT_HERSHEY_COMPLEX):
    """Overlay text with a shadow on the frame."""
    shadow_offset = 2
    shadow_color = (0, 0, 0)
    cv2.putText(frame, text, (position[0]-shadow_offset, position[1]-shadow_offset),
                font_choice, 1.2, shadow_color, 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, font_choice, 1.2, color, 2, cv2.LINE_AA)

def gen_frames():
    """
    Stream MJPEG frames.
    No auto-capture is performed here; instead, the status text prompts the user:
      "In range! Press ENTER to capture"
    """
    global video_feed_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return
    try:
        while video_feed_active:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break

            coverage, _ = calculate_document_coverage(frame)
            display_frame = frame.copy()

            if coverage > MAX_COVERAGE:
                status = "Too close! Move back"
                color = (0, 0, 255)
                font = cv2.FONT_HERSHEY_TRIPLEX
            elif MIN_COVERAGE <= coverage <= MAX_COVERAGE:
                status = "In range! Press ENTER to capture"
                color = (0, 255, 0)
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            else:
                status = f"Move closer ({int(coverage)}% coverage)"
                color = (0, 0, 255)
                font = cv2.FONT_HERSHEY_DUPLEX

            put_styled_text(display_frame, status, (10, 50), color, font)
            cv2.putText(display_frame, f"Coverage: {int(coverage)}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print("ERROR in gen_frames:", e)
        traceback.print_exc()
    finally:
        cap.release()


@app.route('/video_feed')
def video_feed():
    global video_feed_active
    video_feed_active = True  # Reset streaming flag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """
    Capture a single image from the webcam on demand (when the user presses ENTER).
    Returns the captured image as a base64-encoded PNG.
    """
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Webcam could not be opened"}), 500
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({"success": False, "error": "Failed to capture image"}), 500
        ret2, buffer = cv2.imencode('.png', frame)
        if not ret2:
            return jsonify({"success": False, "error": "Image encoding failed"}), 500
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        print("SUCCESS: Image captured via /capture_image")
        return jsonify({"success": True, "image": image_b64})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/stop_video_feed', methods=['POST'])
def stop_video_feed():
    global video_feed_active
    video_feed_active = False
    return jsonify({'status': 'Video feed stopped'})
###############################################################################
# OCR ENDPOINT (unchanged, except for any needed imports)
###############################################################################
@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    if request.method == 'GET':
        captured_file = request.args.get('captured_file')
        if captured_file:
            try:
                with open(captured_file, 'rb') as f:
                    image_data = f.read()
                content_type = 'image/jpeg'
                url = "https://pen-to-print-handwriting-ocr.p.rapidapi.com/recognize/"
                data_payload = {"Session": "string", "includeSubScan": "0"}
                files_payload = {
                    "srcImg": (os.path.basename(captured_file), image_data, content_type)
                }
                headers = {
                    "x-rapidapi-key": "556416b211msh6d16ba5d7e43f47p1a3386jsn6181f7e78f71",
                    "x-rapidapi-host": "pen-to-print-handwriting-ocr.p.rapidapi.com",
                    "Accept": "application/json"
                }
                response = requests.post(url, headers=headers, data=data_payload,
                                         files=files_payload, timeout=200)
                if response.status_code != 200:
                    return jsonify({
                        "error": f"API Error: {response.text}",
                        "success": False
                    }), response.status_code
                ocr_result = response.json()
                extracted_text = ocr_result.get("value", "") or "No text extracted from the document."
                corrected = correct_text(extracted_text)
                return jsonify({"success": True, "text": extracted_text, "source": "captured file"}), 200
            except Exception as e:
                return jsonify({"error": str(e), "success": False}), 500
        else:
            return render_template('ocr.html')

    else:
        try:
            extracted_text = ""
            if 'file' in request.files and request.files['file'].filename:
                file = request.files['file']
                filename = file.filename
                content_type = file.content_type.lower()

                # IMAGE
                if content_type.startswith('image/'):
                    image_data = file.read()
                    url = "https://pen-to-print-handwriting-ocr.p.rapidapi.com/recognize/"
                    data = {"Session": "string", "includeSubScan": "0"}
                    files_payload = {"srcImg": (filename, image_data, content_type)}
                    headers = {
                        "x-rapidapi-key": "0dc8577fe3msh41c216f7d1d454bp112dadjsnd4695f9f82a2",
                        "x-rapidapi-host": "pen-to-print-handwriting-ocr.p.rapidapi.com",
                        "Accept": "application/json"
                    }
                    response = requests.post(url, headers=headers, data=data,
                                             files=files_payload, timeout=200)
                    if response.status_code != 200:
                        return jsonify({
                            "error": f"API Error: {response.text}",
                            "success": False
                        }), response.status_code
                    ocr_result = response.json()
                    extracted_text = ocr_result.get("value", "") or "No text extracted from the document."

                # PDF
                elif content_type == 'application/pdf' or filename.lower().endswith('.pdf'):
                    try:
                        file.seek(0)
                        file_bytes = io.BytesIO(file.read())
                        reader = PdfReader(file_bytes)
                        for page in reader.pages:
                            page_text = page.extract_text() or ""
                            if page_text:
                                extracted_text += page_text + "\n"
                        if not extracted_text.strip():
                            extracted_text = "No text extracted from the PDF."
                    except Exception as e:
                        return jsonify({
                            "error": f"Failed to extract PDF text: {str(e)}",
                            "success": False
                        }), 400

                # MS WORD DOC
                elif content_type in [
                    'application/msword',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                ] or filename.lower().endswith('.doc') or filename.lower().endswith('.docx'):
                    file.seek(0)
                    doc = docx.Document(file)
                    extracted_text = "\n".join(para.text for para in doc.paragraphs)
                    if not extracted_text.strip():
                        extracted_text = "No text extracted from the document."

                else:
                    return jsonify({
                        "error": "Unsupported file type",
                        "success": False
                    }), 400

            elif 'image_data' in request.form:
                # Handling base64 image_data from a form
                base64_data = request.form['image_data']
                header, encoded = base64_data.split(',', 1)
                image_data = base64.b64decode(encoded)
                filename = 'captured-image.png'
                content_type = header.split(';')[0].split(':')[1]

                boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
                CRLF = "\r\n"
                fields = {"Session": "string", "includeSubScan": "0"}
                lines = []
                for key, value in fields.items():
                    lines.append("--" + boundary)
                    lines.append(f'Content-Disposition: form-data; name="{key}"')
                    lines.append("")
                    lines.append(value)
                lines.append("--" + boundary)
                lines.append(f'Content-Disposition: form-data; name="srcImg"; filename="{filename}"')
                lines.append("Content-Type: " + content_type)
                lines.append("")
                body_pre = CRLF.join(lines).encode('utf-8')
                body_post = (CRLF + "--" + boundary + "--" + CRLF).encode('utf-8')
                body = body_pre + CRLF.encode('utf-8') + image_data + body_post

                headers = {
                    "x-rapidapi-key": "983437882emshd61ad345511a31dp1adcc9jsn1daee68b0831",
                    "x-rapidapi-host": "pen-to-print-handwriting-ocr.p.rapidapi.com",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Accept": "application/json"
                }
                conn = http.client.HTTPSConnection("pen-to-print-handwriting-ocr.p.rapidapi.com")
                conn.request("POST", "/recognize/", body, headers)
                res = conn.getresponse()
                data_res = res.read()
                conn.close()
                if res.status != 200:
                    return jsonify({
                        "error": f"API Error: {data_res.decode('utf-8')}",
                        "success": False
                    }), res.status
                ocr_result = json.loads(data_res.decode("utf-8"))
                extracted_text = ocr_result.get("value", "") or "No text extracted from the document."
            else:
                return jsonify({
                    'error': 'No file or image data provided',
                    "success": False
                }), 400

            # Post-processing
            corrected = correct_text(extracted_text)

            # Build analysis prompt
            analysis_prompt = (
                "Please extract the following medical categories from the text: "
                "medicine, workout, diet, disease, symptom. Return the results as JSON "
                "where each key is a category and its value is a list of unique terms found in the text. "
                "Text: " + corrected
            )

            # Make request to model
            analysis_response = requests.post(
                f"{OLLAMA_URL}/v1/completions",
                json={
                    "model": "deepseek-r1:7b",
                    "prompt": analysis_prompt,
                    "max_tokens": 1000,
                    "temperature": 0.3
                },
                timeout=200
            )
            analysis_response.raise_for_status()
            analysis_data = analysis_response.json()
            model_response = analysis_data.get("response", "")
            if not model_response and "choices" in analysis_data and analysis_data["choices"]:
                model_response = analysis_data["choices"][0].get("text", "")

            # Extract JSON from model response
            json_match = re.search(r'\{[\s\S]*\}', model_response)
            if not json_match:
                empty_categories = {
                    "medicine": [],
                    "workout": [],
                    "diet": [],
                    "disease": [],
                    "symptom": []
                }
                return jsonify({
                    "text": extracted_text,
                    "highlighted_text": extracted_text,
                    "categories": empty_categories,
                    "legend": {
                        "medicine": "#ff7f7f",
                        "workout": "#7fbfff",
                        "diet": "#7fff7f",
                        "disease": "#ff7fff",
                        "symptom": "#ffbf7f"
                    },
                    "summary": "No valid summary could be generated.",
                    "warning": "No valid JSON found in model response.",
                    "success": False
                })

            categories = {
                "medicine": [],
                "workout": [],
                "diet": [],
                "disease": [],
                "symptom": []
            }
            try:
                parsed_data = json.loads(json_match.group())
                for cat in categories.keys():
                    if cat in parsed_data and isinstance(parsed_data[cat], list):
                        categories[cat] = list({
                            term.strip() for term in parsed_data[cat]
                            if term.strip() and term.strip().lower() != cat.lower()
                        })
            except Exception as e:
                print("JSON parsing error:", e)

            highlighted_text = highlight_text(extracted_text, categories)
            summary_text = generate_summary(correct_text(extracted_text))
            color_map = {
                "medicine": "#ff7f7f",
                "workout": "#7fbfff",
                "diet": "#7fff7f",
                "disease": "#ff7fff",
                "symptom": "#ffbf7f"
            }

            return jsonify({
                "text": extracted_text,
                "highlighted_text": highlighted_text,
                "categories": categories,
                "legend": color_map,
                "summary": summary_text,
                "success": True
            })
        except Exception as e:
            print(f"Error in OCR endpoint: {str(e)}")
            return jsonify({"error": str(e), "success": False}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)


