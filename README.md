
# Diabetes Management and Risk Assessment System

This project is a Flask-based web application designed for managing diabetes-related data, predicting diabetes risk using machine learning, and generating detailed reports. The application includes functionalities for doctors to register, log in, manage patient data, and perform analytics. Additionally, it leverages SMS-based OTP verification for secure access.

## Features

### User Management

- Doctor Registration & Login: Includes phone number verification using OTP.
- Profile Management: Update doctor details like name, email, phone, and specialization.
- Password Management: Reset and change passwords securely.

### Data Management
- Upload CSV Files: Bulk insert patient data.
- View & Filter Data: Paginated and sortable views with filtering options.
- Download CSV: Export filtered data as CSV.

### Analytics
- Generate insights from patient data, such as:
    - Age, BMI, and glucose distributions.
    - Diabetes risk predictions.
    - Correlation analysis between features.
- Visualize results using charts powered by Chart.js.

### Diabetes Prediction
- Predict diabetes risk using a pre-trained Support Vector Classifier (SVC).
- Confidence scores and visualizations of prediction probabilities.

### Reporting
- Generate PDF reports with patient details, risk assessments, and recommendations.

## Prerequisites
### Software
- Python 3.11 or above
- Flask and required libraries (see requirements.txt)
- SQLite for the database
- Twilio for SMS-based OTP verification

### Environment Variables
Set up the following variables in a .env file:

    SECRET_KEY
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_VERIFY_SID

## Setup Instructions

1. Clone the Repository

```
git clone https://github.com/rahulgowdaa/Diabetes-Prediction-Web-Application.git
cd Diabetes-Prediction-Web-Application
```
2. Install Dependencies

```
pip install -r requirements.txt
```
3. Configure Environment Variables Create a .env file in the root directory and add:
```
SECRET_KEY=<your-secret-key>
TWILIO_ACCOUNT_SID=<your-twilio-account-sid>
TWILIO_AUTH_TOKEN=<your-twilio-auth-token>
TWILIO_VERIFY_SID=<your-twilio-verify-sid>
```
4. Run the Application
```
python app.py
```
The application will be available at http://127.0.0.1:5000/

## Usage
- Register and Verify: Doctors register and verify their phone numbers using OTP.
- Login: Access the dashboard after successful authentication.
- Upload Data: Insert patient data manually or via CSV upload.
- Analyze Data: View analytics dashboards and generate predictions.
- Generate Reports: Create PDF reports for patients.

## Directory Structure
```bash
├── flask_app/                  # Main application folder
│   ├── __pycache__/            # Python bytecode cache
│   ├── static/css/             # Static assets
│   │   └── style.css           # Custom CSS for styling
│   ├── templates/              # HTML templates
│   │   ├── analytics.html
│   │   ├── base.html           # Base layout
│   │   ├── change_password.html
│   │   ├── display.html
│   │   ├── doctor_profile.html
│   │   ├── forgot_password.html
│   │   ├── form.html
│   │   ├── index.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── reset_password.html
│   │   ├── result.html
│   │   └── verify_otp.html
│   ├── app.py                  # Main Flask application
│   ├── train_model.py          # Script to train the machine learning model
├── models/                     # Pre-trained ML models
│   ├── diabetes_model.pkl      # SVM model for diabetes prediction
│   └── scaler.pkl              # Scaler for feature normalization
├── venv/                       # Python virtual environment
├── .env                        # Environment variables
├── diabetes.db                 # SQLite database
├── .gitignore                  # Git ignore file
├── LICENSE                     # License information
└── README.md                   # Project documentation
``` 
## License
This project is licensed under the MIT License. 