import joblib
from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # Initializing a Flask app

@app.route('/', methods=['GET'])  # Home Page
def homePage():
    return render_template("index.html")


@app.route('/train', methods=['GET'])  # Train Model Route
def training():
    os.system("python main.py")
    return "Training Successful!"


@app.route('/predict', methods=['POST'])  # Ensure only `POST` requests
def predict():
    try:
        print(request.form)

        # Reading input from form
        input_data = {
            "customerID": request.form.get('customerID', ''),
            "gender": request.form.get('gender', 'Male'),
            "SeniorCitizen": int(request.form.get('SeniorCitizen', 0)),
            "Partner": request.form.get('Partner', 'No'),
            "Dependents": request.form.get('Dependents', 'No'),
            "tenure": int(request.form.get('tenure', 0)),
            "PhoneService": request.form.get('PhoneService', 'No'),
            "MultipleLines": request.form.get('MultipleLines', 'No phone service'),
            "InternetService": request.form.get('InternetService', 'No'),
            "OnlineSecurity": request.form.get('OnlineSecurity', 'No internet service'),
            "OnlineBackup": request.form.get('OnlineBackup', 'No internet service'),
            "DeviceProtection": request.form.get('DeviceProtection', 'No internet service'),
            "TechSupport": request.form.get('TechSupport', 'No internet service'),
            "StreamingTV": request.form.get('StreamingTV', 'No internet service'),
            "StreamingMovies": request.form.get('StreamingMovies', 'No internet service'),
            "Contract": request.form.get('Contract', 'Month-to-month'),
            "PaperlessBilling": request.form.get('PaperlessBilling', 'No'),
            "PaymentMethod": request.form.get('PaymentMethod', 'Electronic check'),
            "MonthlyCharges": float(request.form.get('MonthlyCharges', 0.0)),
            "TotalCharges": request.form.get('TotalCharges', '0.0')
        }
        print(input_data)
        # Initialize PredictionPipeline
        predictor = PredictionPipeline()

        # Get prediction
        prediction = predictor.predict(input_data)

        # Interpret prediction output
        churn_result = "Yes, customer will churn" if prediction[0] == 1 else "No, customer will not churn"

        return render_template('results.html', prediction=churn_result)

    except Exception as e:
        print(f'Error: {e}')
        return 'Something went wrong. Please try again.'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
