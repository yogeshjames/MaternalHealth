from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Enable CORS (for handling cross-origin requests)
CORS(app)

# Load the trained Random Forest model and label encoder
model = joblib.load('random_forest_model.pkl')  # Ensure this is the correct path
label_encoder = joblib.load('label_encoder.pkl')  # Ensure this is the correct path


@app.route('/')
def home():
    return "Maternal Health Risk Prediction API"


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Extract the features from the data
    try:
        features = [
            data['Age'],
            data['Systolic BP'],
            data['Diastolic'],
            data['BS'],
            data['Body Temp'],
            data['BMI'],
            data['Previous Complications'],
            data['Preexisting Diabetes'],
            data['Gestational Diabetes'],
            data['Mental Health'],
            data['Heart Rate']
        ]

        # Convert features to a NumPy array for prediction
        input_data = np.array([features])

        # Make the prediction using the model
        prediction = model.predict(input_data)

        # Inverse transform the label to get the actual risk level
        risk_level = label_encoder.inverse_transform([prediction])[0]

        # Return the prediction as JSON response
        return jsonify({'prediction': risk_level})

    except Exception as e:
        # Handle any error that occurs during prediction
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)