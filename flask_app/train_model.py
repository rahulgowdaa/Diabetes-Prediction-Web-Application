import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import os
import sqlite3

def train_and_save_model():
    try:
        # Connect to your SQLite database
        conn = sqlite3.connect('diabetes.db')
        
        # Read data from database
        query = """
            SELECT frontfoot, rearfoot, Glucose, BloodPressure, 
                   Insulin, BMI, midfootavg, age, Outcome 
            FROM diabetes 
            WHERE Outcome IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        model = SVC(kernel='linear', C=10.0, probability=True)
        model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.2f}")
        print(f"Testing Accuracy: {test_accuracy:.2f}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save the model and scaler
        joblib.dump(model, 'models/diabetes_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        print("Model and scaler saved successfully!")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        return False

if __name__ == "__main__":
    train_and_save_model() 