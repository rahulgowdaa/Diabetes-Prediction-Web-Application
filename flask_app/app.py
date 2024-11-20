from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response, make_response, jsonify
import sqlite3
import pandas as pd
from reportlab.pdfgen import canvas
import os
from jinja2 import Environment
import csv 
import io
from fpdf import FPDF
import re
from datetime import datetime
from flask_cors import CORS
from sklearn.svm import SVC
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from twilio.rest import Client
import random
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from functools import wraps

from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.jinja_env.globals.update(max=max, min=min)

# Update Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_VERIFY_SID = os.getenv('TWILIO_VERIFY_SID')
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

CORS(app)
# Initialize the SQLite database and create the table if it doesn't exist
def init_db():
    try:
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        # Drop the table if it exists and recreate it
        cursor.execute('DROP TABLE IF EXISTS diabetes')
        
        # Create new table with all required fields
        cursor.execute('''
            CREATE TABLE diabetes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                dob DATE,
                address TEXT,
                frontfoot INTEGER,
                rearfoot INTEGER,
                Glucose INTEGER,
                BloodPressure INTEGER,
                Insulin INTEGER,
                BMI REAL,
                midfootavg REAL,
                Outcome INTEGER
            )
        ''')
        
        # Create doctors table with phone and verification fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT UNIQUE NOT NULL,
                specialization TEXT,
                is_verified BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create OTP table for verification
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctor_otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER,
                otp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_used BOOLEAN DEFAULT 0,
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully!")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        conn.close()

# Add this login check decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'doctor_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Route to download filtered data as CSV
@app.route('/download_csv')
@login_required
def download_csv():
    filter_by = request.args.get('filter_by', '')
    filter_value = request.args.get('filter_value', '')
    sort_by = request.args.get('sort_by', 'id')
    order = request.args.get('order', 'asc')

    # Fetch filtered, sorted data
    data, _ = get_filtered_sorted_paginated_data(1, 1000, sort_by, order, filter_by, filter_value)

    # Create CSV in-memory file
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)

    # Write headers
    writer.writerow(['ID', 'Name', 'Age', 'DOB', 'Address', 'Front Foot', 'Rear Foot', 'Glucose', 'Blood Pressure', 'Insulin', 'BMI', 'Mid Foot Avg', 'Outcome'])
    
    # Write data rows
    for row in data:
        writer.writerow(row)

    # Set CSV data for download
    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers.set("Content-Disposition", "attachment", filename="filtered_data.csv")
    return response

@app.route('/insert_csv', methods=['POST'])
def insert_csv():
    if 'file' not in request.files:
        flash("No file part", "danger")
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        flash("No selected file", "danger")
        return redirect(url_for('index'))

    if file and file.filename.endswith('.csv'):
        try:
            # Read the CSV file
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            reader = csv.reader(stream)

            # Check CSV headers
            required_headers = ['id', 'frontfoot', 'Glucose', 'BloodPressure', 'rearfoot', 'Insulin', 'BMI', 'midfootavg', 'Age', 'Outcome']
            headers = next(reader)  # First row is the header

            # Verify all required headers are present
            missing_headers = [h for h in required_headers if h not in headers]
            if missing_headers:
                flash(f"Missing required headers: {', '.join(missing_headers)}", "danger")
                return redirect(url_for('index'))

            # Ensure database and table exist
            init_db()

            conn = sqlite3.connect('diabetes.db')
            cursor = conn.cursor()

            inserted_rows = 0
            duplicate_rows = 0

            for row in reader:
                # Rest of your existing code...
                row_dict = dict(zip(headers, row))
                id = row_dict.get('id')
                
                # Check if this ID already exists
                cursor.execute("SELECT 1 FROM diabetes WHERE id = ?", (id,))
                if cursor.fetchone():
                    duplicate_rows += 1
                    continue

                # Prepare data with N/A for missing personal info
                data = (
                    id,
                    'N/A',  # name
                    row_dict.get('Age', 0),  # age
                    'N/A',  # dob
                    'N/A',  # address
                    row_dict.get('frontfoot'),
                    row_dict.get('rearfoot'),
                    row_dict.get('Glucose'),
                    row_dict.get('BloodPressure'),
                    row_dict.get('Insulin'),
                    row_dict.get('BMI'),
                    row_dict.get('midfootavg'),
                    row_dict.get('Outcome')
                )

                try:
                    cursor.execute('''
                        INSERT INTO diabetes (
                            id, name, age, dob, address, 
                            frontfoot, rearfoot, Glucose, BloodPressure, 
                            Insulin, BMI, midfootavg, Outcome
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', data)
                    inserted_rows += 1
                except sqlite3.IntegrityError as e:
                    flash(f"Error inserting row with ID {id}: {e}", "danger")

            conn.commit()
            conn.close()

            flash(f"Inserted {inserted_rows} rows. Skipped {duplicate_rows} duplicate rows.", "success")
            return redirect(url_for('index'))

        except Exception as e:
            flash(f"Error processing CSV file: {e}", "danger")
            return redirect(url_for('index'))

    flash("Invalid file format. Please upload a CSV file.", "danger")
    return redirect(url_for('index'))

# Home Page (Navbar)
@app.route('/')
def index():
    if 'doctor_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Route to display the form page
@app.route('/form', methods=['GET', 'POST'])
@login_required
def form_submit():
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            age = int(request.form['age'])
            dob = request.form['dob']
            address = request.form['address']
            frontfoot = float(request.form['frontfoot'])
            rearfoot = float(request.form['rearfoot'])
            glucose = float(request.form['Glucose'])
            blood_pressure = float(request.form['BloodPressure'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            midfootavg = float(request.form['midfootavg'])
            
            # Load model and scaler
            model, scaler = load_model_and_scaler()
            
            # Create features DataFrame with column names
            feature_names = ['frontfoot', 'rearfoot', 'Glucose', 'BloodPressure',
                           'Insulin', 'BMI', 'midfootavg', 'age']
            features = pd.DataFrame([[frontfoot, rearfoot, glucose, blood_pressure,
                                    insulin, bmi, midfootavg, age]], 
                                  columns=feature_names)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            confidence = max(prediction_proba) * 100
            
            # Generate PDF if requested
            if 'generate_pdf' in request.form:
                return generate_patient_report(
                    name, age, dob, address, frontfoot, rearfoot,
                    glucose, blood_pressure, insulin, bmi, midfootavg,
                    prediction
                )
            
            # Insert into database
            conn = sqlite3.connect('diabetes.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO diabetes (
                    name, age, dob, address, frontfoot, rearfoot,
                    Glucose, BloodPressure, Insulin, BMI, midfootavg, Outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, age, dob, address, frontfoot, rearfoot,
                  glucose, blood_pressure, insulin, bmi, midfootavg, int(prediction)))
            
            conn.commit()
            conn.close()
            
            # Render result template
            return render_template('result.html',
                                name=name, age=age, dob=dob, address=address,
                                frontfoot=frontfoot, rearfoot=rearfoot,
                                glucose=glucose, blood_pressure=blood_pressure,
                                insulin=insulin, bmi=bmi, midfootavg=midfootavg,
                                prediction=prediction, confidence=confidence,
                                pdf_generated='generate_pdf' in request.form)
            
        except Exception as e:
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('form_submit'))
            
    return render_template('form.html')

# Function to generate PDF
def generate_patient_report(name, age, dob, address, frontfoot, rearfoot, glucose, blood_pressure, insulin, bmi, midfootavg, outcome):
    pdf = FPDF()
    pdf.add_page()

    # Add header with logo (if you have one)
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(200, 20, txt="Diabetes Risk Assessment Report", ln=True, align='C')
    
    # Add date and reference number
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=True, align='R')
    pdf.cell(200, 10, txt=f"Reference: PAT-{datetime.now().strftime('%Y%m%d%H%M')}", ln=True, align='R')
    
    # Patient Information Section
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 15, txt="Patient Information", ln=True)
    
    # Create a table-like structure for patient details
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(44, 62, 80)
    details = [
        ["Name", name],
        ["Age", f"{age} years"],
        ["Date of Birth", dob],
        ["Address", address]
    ]
    
    for detail in details:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(50, 10, txt=detail[0] + ":", ln=0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(150, 10, txt=str(detail[1]), ln=1)
    
    # Medical Measurements Section
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 15, txt="Medical Measurements", ln=True)
    
    # Create measurement table
    measurements = [
        ["Glucose Level", f"{glucose} mg/dL", "Normal Range: 70-140 mg/dL"],
        ["Blood Pressure", f"{blood_pressure} mmHg", "Normal Range: 90-120 mmHg"],
        ["Insulin Level", f"{insulin} uU/mL", "Normal Range: 2-25 uU/mL"],
        ["BMI", f"{bmi:.1f}", "Normal Range: 18.5-24.9"],
    ]
    
    for measure in measurements:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(50, 10, txt=measure[0] + ":", ln=0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(50, 10, txt=str(measure[1]), ln=0)
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(100, 10, txt=measure[2], ln=1)
        pdf.set_text_color(44, 62, 80)
    
    # Foot Pressure Analysis
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 15, txt="Foot Pressure Analysis", ln=True)
    
    foot_measurements = [
        ["Front Foot Pressure", f"{frontfoot} kPa"],
        ["Rear Foot Pressure", f"{rearfoot} kPa"],
        ["Mid Foot Average", f"{midfootavg} kPa"]
    ]
    
    for measure in foot_measurements:
        pdf.set_font('Arial', 'B', 11)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(70, 10, txt=measure[0] + ":", ln=0)
        pdf.set_font('Arial', '', 11)
        pdf.cell(130, 10, txt=str(measure[1]), ln=1)
    
    # Risk Assessment
    pdf.ln(15)
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 15, txt="Risk Assessment Result", ln=True)
    
    # Risk level with color-coded box
    risk_level = 'High Risk' if outcome == 1 else 'Low Risk'
    risk_color = (192, 57, 43) if outcome == 1 else (39, 174, 96)
    
    pdf.set_fill_color(*risk_color)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 20, txt=f"Diabetes Risk Level: {risk_level}", ln=True, align='C', fill=True)
    
    # Recommendations
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(200, 15, txt="Recommendations", ln=True)
    
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(44, 62, 80)
    recommendations = [
        "* Regular monitoring of blood glucose levels",
        "* Maintain a balanced diet and regular exercise routine",
        "* Regular check-ups with healthcare provider",
        "* Monitor foot health and pressure distribution",
        "* Keep track of blood pressure and insulin levels"
    ]
    
    for rec in recommendations:
        pdf.cell(200, 10, txt=rec, ln=True)
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(190, 5, txt="Disclaimer: This report is generated based on the provided measurements and should be used for reference only. Please consult with a healthcare professional for proper medical advice and diagnosis.", align='L')
    
    # Save to a temporary file
    temp_pdf_path = f"temp_{name}_report.pdf"
    pdf.output(temp_pdf_path)
    
    # Read the file and create response
    with open(temp_pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    # Delete the temporary file
    os.remove(temp_pdf_path)
    
    # Create response
    response = make_response(pdf_content)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={name}_diabetes_report.pdf'
    
    return response

# Function to get filtered, sorted, and paginated data from the database
def get_filtered_sorted_paginated_data(page, per_page, sort_by, order, filter_by, filter_value):
    conn = sqlite3.connect('diabetes.db')
    cursor = conn.cursor()

    query = "SELECT * FROM diabetes"
    params = []

    # Apply filtering
    if filter_by and filter_value:
        query += f" WHERE {filter_by} LIKE ?"
        params.append(f"%{filter_value}%")

    # Apply sorting
    if sort_by:
        query += f" ORDER BY {sort_by} {order}"

    # Pagination logic
    offset = (page - 1) * per_page
    query += " LIMIT ? OFFSET ?"
    params.extend([per_page, offset])

    cursor.execute(query, params)
    data = cursor.fetchall()

    # Get the total number of rows for pagination
    cursor.execute("SELECT COUNT(*) FROM diabetes")
    total_rows = cursor.fetchone()[0]

    conn.close()
    return data, total_rows

# Route to display data with sort, filter, and pagination functionality
@app.route('/display')
@login_required
def display():
    # Get sorting, filtering, and pagination options from query parameters
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    sort_by = request.args.get('sort_by', 'id')  # Default sort by 'id'
    order = request.args.get('order', 'asc')  # Default order 'asc'
    filter_by = request.args.get('filter_by', '')
    filter_value = request.args.get('filter_value', '')

    # Fetch data based on the parameters (this function will need to handle filtering/sorting)
    data, total_rows = get_filtered_sorted_paginated_data(page, per_page, sort_by, order, filter_by, filter_value)

    # Pagination logic
    total_pages = (total_rows + per_page - 1) // per_page
    max_display_pages = 5
    start_page = max(1, page - max_display_pages // 2)
    end_page = min(total_pages, start_page + max_display_pages - 1)

    return render_template('display.html', data=data, page=page, per_page=per_page, total_pages=total_pages,
                           start_page=start_page, end_page=end_page, sort_by=sort_by, order=order, 
                           filter_by=filter_by, filter_value=filter_value)

@app.route('/init-db')
def initialize_database():
    try:
        init_db()
        flash("Database initialized successfully!", "success")
    except Exception as e:
        flash(f"Error initializing database: {e}", "danger")
    return redirect(url_for('index'))

@app.route('/analytics')
def analytics():
    try:
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        # Debug print 1
        print("Starting analytics function")
        
        # Check if there's any data
        cursor.execute("SELECT COUNT(*) FROM diabetes")
        total_count = cursor.fetchone()[0]
        if total_count == 0:
            flash("No data available for analytics. Please add some records first.", "warning")
            return redirect(url_for('index'))

        # Fetch data for analytics
        cursor.execute("""
            SELECT age, BMI, Glucose, BloodPressure, Insulin, Outcome,
                   frontfoot, rearfoot, midfootavg
            FROM diabetes
            WHERE age IS NOT NULL 
              AND BMI IS NOT NULL 
              AND Glucose IS NOT NULL
              AND BloodPressure IS NOT NULL
              AND Insulin IS NOT NULL
              AND Outcome IS NOT NULL
        """)
        results = cursor.fetchall()
        conn.close()

        # Debug print 2
        print("Data fetched from database")
        
        if not results:
            flash("Insufficient data for analytics. Please add complete records.", "warning")
            return redirect(url_for('index'))

        # Convert to DataFrame
        columns = ['age', 'BMI', 'Glucose', 'BloodPressure', 'Insulin', 
                  'Outcome', 'frontfoot', 'rearfoot', 'midfootavg']
        df = pd.DataFrame(results, columns=columns)
        
        # Debug print 3
        print("DataFrame created")

        # Basic statistics - explicitly convert all values
        basic_stats = {
            'total_patients': int(total_count),
            'averages': {
                'bmi': float(df['BMI'].mean()),
                'glucose': float(df['Glucose'].mean()),
                'bp': float(df['BloodPressure'].mean()),
                'insulin': float(df['Insulin'].mean())
            }
        }

        # Convert series to lists with explicit type conversion
        chart_data = {
            'age_data': [float(x) for x in df['age'].values],
            'bmi_data': [float(x) for x in df['BMI'].values],
            'glucose_data': [float(x) for x in df['Glucose'].values],
            'bp_data': [float(x) for x in df['BloodPressure'].values],
            'insulin_data': [float(x) for x in df['Insulin'].values],
            'diabetes_counts': {
                'Diabetic': int(df['Outcome'].sum()),
                'Non-Diabetic': int(len(df) - df['Outcome'].sum())
            }
        }

        # Debug print 4
        print("Basic statistics calculated")

        # Model features
        feature_cols = ['frontfoot', 'rearfoot', 'Glucose', 'BloodPressure', 
                       'Insulin', 'BMI', 'midfootavg', 'age']
        X = df[feature_cols]
        y = df['Outcome']

        # Load model and make predictions
        model, scaler = load_model_and_scaler()
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        
        # Calculate ROC curve values
        fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        # Calculate feature importance
        if hasattr(model, 'coef_'):
            feature_importance_values = [float(x) for x in np.abs(model.coef_[0])]
        else:
            feature_importance_values = [1.0] * len(feature_cols)

        # Calculate correlation matrix
        correlation_matrix = df[feature_cols].corr()
        correlation_values = []
        for i in range(len(feature_cols)):
            row = []
            for j in range(len(feature_cols)):
                row.append(float(correlation_matrix.iloc[i, j]))
            correlation_values.append(row)

        # Prepare template data with explicit type conversion
        template_data = {
            'age_data': [float(x) for x in df['age'].values],
            'bmi_data': [float(x) for x in df['BMI'].values],
            'glucose_data': [float(x) for x in df['Glucose'].values],
            'bp_data': [float(x) for x in df['BloodPressure'].values],
            'insulin_data': [float(x) for x in df['Insulin'].values],
            'total_patients': int(len(df)),
            'averages': {
                'bmi': float(df['BMI'].mean()),
                'glucose': float(df['Glucose'].mean()),
                'bp': float(df['BloodPressure'].mean()),
                'insulin': float(df['Insulin'].mean())
            },
            'diabetes_counts': {
                'Diabetic': int(df['Outcome'].sum()),
                'Non-Diabetic': int(len(df) - df['Outcome'].sum())
            },
            'model_metrics': {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred)),
                'recall': float(recall_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred))
            },
            'roc_curve': {
                'fpr': [float(x) for x in fpr],
                'tpr': [float(x) for x in tpr],
                'auc': float(roc_auc)
            },
            'feature_importance_names': [str(x) for x in feature_cols],  # Convert to strings
            'feature_importance_values': feature_importance_values,
            'correlation_matrix_labels': [str(x) for x in feature_cols],
            'correlation_matrix_values': correlation_values,
            'prediction_confidence_dist': [float(x) for x in np.histogram(
                np.max(y_pred_proba, axis=1),
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
            )[0]]
        }

        # Test JSON serialization before rendering
        import json
        try:
            json.dumps(template_data)
        except TypeError as e:
            print(f"JSON serialization test failed: {str(e)}")
            raise

        return render_template('analytics.html', **template_data)

    except Exception as e:
        import traceback
        print(f"Analytics Error: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        flash(f"Error generating analytics: {str(e)}", "danger")
        return redirect(url_for('index'))

# Add this function to load model and scaler
def load_model_and_scaler():
    try:
        model = joblib.load('models/diabetes_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        from train_model import train_and_save_model
        if train_and_save_model():
            return joblib.load('models/diabetes_model.pkl'), joblib.load('models/scaler.pkl')
        else:
            raise Exception("Failed to train and load model")

def generate_otp():
    return ''.join([str(random.randint(0, 9)) for _ in range(6)])

def send_otp(phone_number, otp):
    try:
        # Ensure phone number is in E.164 format
        if not phone_number.startswith('+'):
            # Add India's country code (+91) if not present
            phone_number = '+91' + phone_number.lstrip('0')
            
        verification = twilio_client.verify \
            .v2 \
            .services(TWILIO_VERIFY_SID) \
            .verifications \
            .create(to=phone_number, channel='sms')
        return True
    except Exception as e:
        print(f"Error sending SMS: {str(e)}")
        return False

def verify_otp(phone_number, code):
    try:
        # Ensure phone number is in E.164 format
        if not phone_number.startswith('+'):
            # Add India's country code (+91) if not present
            phone_number = '+91' + phone_number.lstrip('0')
            
        verification_check = twilio_client.verify \
            .v2 \
            .services(TWILIO_VERIFY_SID) \
            .verification_checks \
            .create(to=phone_number, code=code)
        return verification_check.status == 'approved'
    except Exception as e:
        print(f"Error verifying OTP: {str(e)}")
        return False

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        specialization = request.form.get('specialization')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        try:
            conn = sqlite3.connect('diabetes.db')
            cursor = conn.cursor()
            
            # Hash password
            hashed_password = generate_password_hash(password)
            
            # Set is_verified to 0 initially
            cursor.execute('''
                INSERT INTO doctors (username, password_hash, name, email, phone, specialization, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, hashed_password, name, email, phone, specialization, 0))
            
            doctor_id = cursor.lastrowid
            conn.commit()
            
            # Send OTP
            if send_otp(phone, None):  # We don't need to pass OTP as Twilio Verify handles it
                session['pending_verification'] = doctor_id
                flash('Please verify your phone number', 'info')
                return redirect(url_for('verify_otp_route'))
            else:
                # If OTP sending fails, delete the user and show error
                cursor.execute('DELETE FROM doctors WHERE id = ?', (doctor_id,))
                conn.commit()
                flash('Failed to send verification code', 'danger')
                return redirect(url_for('register'))
            
        except sqlite3.Error as e:
            flash(f'Error during registration: {str(e)}', 'danger')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp_route():
    if 'pending_verification' not in session:
        return redirect(url_for('register'))
    
    if request.method == 'POST':
        otp_code = request.form.get('otp')
        doctor_id = session['pending_verification']
        
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        try:
            # Get doctor's phone number
            cursor.execute('SELECT phone FROM doctors WHERE id = ?', (doctor_id,))
            result = cursor.fetchone()
            
            if not result:
                flash('User not found', 'danger')
                return redirect(url_for('register'))
                
            phone_number = result[0]
            
            if verify_otp(phone_number, otp_code):
                # Mark doctor as verified
                cursor.execute('UPDATE doctors SET is_verified = 1 WHERE id = ?', (doctor_id,))
                conn.commit()
                
                session.pop('pending_verification', None)
                flash('Phone number verified! You can now login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Invalid or expired verification code', 'danger')
        
        except Exception as e:
            flash(f'Error during verification: {str(e)}', 'danger')
        finally:
            conn.close()
    
    return render_template('verify_otp.html')

@app.route('/resend-otp', methods=['POST'])
def resend_otp():
    if 'pending_verification' not in session:
        return jsonify({'success': False, 'message': 'No pending verification'})
    
    doctor_id = session['pending_verification']
    
    try:
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        # Get doctor's phone number
        cursor.execute('SELECT phone FROM doctors WHERE id = ?', (doctor_id,))
        phone = cursor.fetchone()[0]
        
        # Generate and store new OTP
        otp = generate_otp()
        cursor.execute('''
            INSERT INTO doctor_otps (doctor_id, otp, expires_at)
            VALUES (?, ?, datetime('now', '+15 minutes'))
        ''', (doctor_id, otp))
        
        conn.commit()
        
        # Send new OTP
        if send_otp(phone, otp):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Error sending OTP'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, password_hash, name, is_verified 
            FROM doctors WHERE username = ?
        ''', (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user[1], password):
            if not user[3]:  # is_verified check
                flash('Please verify your phone number first', 'warning')
                session['pending_verification'] = user[0]
                return redirect(url_for('verify_otp_route'))
                
            session['doctor_id'] = user[0]
            session['doctor_name'] = user[2]
            flash(f'Welcome back, Dr. {user[2]}!', 'success')
            return redirect(url_for('index'))
        
        flash('Invalid username or password', 'danger')
        conn.close()
    
    return render_template('login.html')

@app.route('/home')
def home():
    if 'doctor_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        try:
            # Check if email exists
            cursor.execute('SELECT id, phone FROM doctors WHERE email = ?', (email,))
            user = cursor.fetchone()
            
            if user:
                doctor_id, phone = user
                # Send OTP
                if send_otp(phone, None):
                    session['reset_password_id'] = doctor_id
                    flash('Verification code sent to your phone', 'success')
                    return redirect(url_for('reset_password'))
                else:
                    flash('Error sending verification code', 'danger')
            else:
                flash('No account found with that email address', 'danger')
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
        finally:
            conn.close()
            
    return render_template('forgot_password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_password_id' not in session:
        return redirect(url_for('forgot_password'))
        
    if request.method == 'POST':
        otp = request.form.get('otp')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('reset_password'))
            
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        try:
            # Get user's phone number
            cursor.execute('SELECT phone FROM doctors WHERE id = ?', (session['reset_password_id'],))
            phone = cursor.fetchone()[0]
            
            # Verify OTP
            if verify_otp(phone, otp):
                # Update password
                hashed_password = generate_password_hash(new_password)
                cursor.execute('''
                    UPDATE doctors 
                    SET password_hash = ? 
                    WHERE id = ?
                ''', (hashed_password, session['reset_password_id']))
                
                conn.commit()
                session.pop('reset_password_id', None)
                flash('Password has been reset successfully', 'success')
                return redirect(url_for('login'))
            else:
                flash('Invalid verification code', 'danger')
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
        finally:
            conn.close()
            
    return render_template('reset_password.html')

@app.route('/doctor-profile', methods=['GET', 'POST'])
@login_required
def doctor_profile():
    conn = sqlite3.connect('diabetes.db')
    cursor = conn.cursor()
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        specialization = request.form.get('specialization')
        
        try:
            cursor.execute('''
                UPDATE doctors 
                SET name = ?, email = ?, phone = ?, specialization = ?
                WHERE id = ?
            ''', (name, email, phone, specialization, session['doctor_id']))
            
            conn.commit()
            session['doctor_name'] = name  # Update session name
            flash('Profile updated successfully!', 'success')
            
        except sqlite3.Error as e:
            flash(f'Error updating profile: {str(e)}', 'danger')
            
    # Get current doctor's details
    cursor.execute('''
        SELECT username, name, email, phone, specialization, created_at 
        FROM doctors WHERE id = ?
    ''', (session['doctor_id'],))
    
    doctor = cursor.fetchone()
    conn.close()
    
    return render_template('doctor_profile.html', doctor=doctor)

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return redirect(url_for('change_password'))
            
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT password_hash FROM doctors WHERE id = ?', (session['doctor_id'],))
            stored_hash = cursor.fetchone()[0]
            
            if check_password_hash(stored_hash, current_password):
                hashed_password = generate_password_hash(new_password)
                cursor.execute('''
                    UPDATE doctors 
                    SET password_hash = ? 
                    WHERE id = ?
                ''', (hashed_password, session['doctor_id']))
                
                conn.commit()
                flash('Password changed successfully', 'success')
                return redirect(url_for('doctor_profile'))
            else:
                flash('Current password is incorrect', 'danger')
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
        finally:
            conn.close()
            
    return render_template('change_password.html')

# Modify your main block
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
