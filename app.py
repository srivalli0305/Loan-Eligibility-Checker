from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define columns for input data
input_columns = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

@app.route('/')
def home():
    return render_template('loan_eligibility.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received POST request")
        form_data = request.get_json()
        print(f"Form data: {form_data}")

        # Validate required inputs
        required_fields = ['Gender', 'Married', 'Education', 'ApplicantIncome', 'Credit_History', 'Property_Area']
        for field in required_fields:
            if not form_data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Convert inputs to DataFrame
        df = pd.DataFrame([form_data], columns=input_columns)

        # Convert numeric fields to appropriate types
        numeric_fields = [
            'Dependents', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History'
        ]
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)

        # Encode categorical variables
        categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = label_encoders[col].transform(df[col].astype(str))

        # Make prediction
        prediction = model.predict(df)
        result = "Eligible for Loan" if prediction[0] == 1 else "Not Eligible for Loan"

        print(f"Prediction result: {result}")
        return jsonify({"eligibility": result})

    except Exception as e:
        print(f"Error: {e}")  # Log the error
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True)
