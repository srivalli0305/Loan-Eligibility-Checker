import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib 

# Load dataset
data = pd.read_csv("loan_dataset.csv")

# Drop Loan_ID (or any other irrelevant column)
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True) 
data.fillna(data.mode().iloc[0], inplace=True)          

# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data into features (X) and target (y)
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")

# Save the encoders for preprocessing in the Flask app
joblib.dump(label_encoders, "label_encoders.pkl")
