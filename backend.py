# backend.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app) 

# --- Load The Trained Models ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    price_model_path = os.path.join(script_dir, 'app', 'house_price_model.pkl')
    style_model_path = os.path.join(script_dir, 'app', 'house_style_model.pkl')

    price_model = joblib.load(price_model_path)
    style_model = joblib.load(style_model_path)
    print("✅ Models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    price_model = None
    style_model = None

# --- Route to serve the HTML frontend ---
@app.route('/')
def home():
    return render_template('index.html')

# --- API Endpoint for prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    if not price_model or not style_model:
        return jsonify({'error': 'Models are not loaded properly.'}), 500

    try:
        # Get the data sent from the HTML form
        data = request.get_json()

        # **THE FIX IS HERE**: Convert all incoming data to the correct numeric types
        # This prevents errors if the JavaScript sends numbers as strings.
        input_data_dict = {
            "LotArea": [float(data['lot_area'])],
            "OverallQual": [int(data['overall_qual'])],
            "YearBuilt": [int(data['year_built'])],
            "GrLivArea": [float(data['gr_liv_area'])],
            "FullBath": [int(data['full_bath'])],
            "BedroomAbvGr": [int(data['bedroom_abvgr'])],
            "GarageCars": [int(data['garage_cars'])]
        }
        
        input_data = pd.DataFrame(input_data_dict)

        # Get predictions from both models
        price_prediction = price_model.predict(input_data)[0]
        style_prediction = style_model.predict(input_data)[0]
        
        # Get feature importances from the price model
        importances = price_model.feature_importances_
        feature_names = input_data.columns
        
        importance_list = [{'feature': f, 'value': v} for f, v in zip(feature_names, importances)]

        # Send the results back to the frontend as JSON
        return jsonify({
            'price': price_prediction,
            'style': style_prediction,
            'importances': importance_list
        })
    
    except Exception as e:
        # If any error occurs during prediction, send it back for debugging
        print(f"❌ Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# --- Run the Backend Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
