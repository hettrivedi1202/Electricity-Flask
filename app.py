from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define the feature columns in the exact order used during training
feature_columns = [
    'site area',
    'water consumption',
    'recycling rate',
    'utilisation rate',
    'air qality index',
    'issue reolution time',
    'resident count',
    'structure type_Industrial',
    'structure type_Mixed-use',
    'structure type_Residential'
]

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        site_area = float(request.form['site_area'])
        water_consumption = float(request.form['water_consumption'])
        recycling_rate = float(request.form['recycling_rate'])
        utilisation_rate = float(request.form['utilisation_rate'])
        aqi = float(request.form['aqi'])
        issue_time = float(request.form['issue_time'])
        resident_count = float(request.form['resident_count'])
        
        structure_type = request.form['structure_type']
        
        # One-hot encode the structure type
        industrial = 1 if structure_type == 'Industrial' else 0
        mixed = 1 if structure_type == 'Mixed-use' else 0
        residential = 1 if structure_type == 'Residential' else 0
        
        # Create DataFrame with correct column order
        input_data = pd.DataFrame([[site_area, 
                                   water_consumption, 
                                   recycling_rate, 
                                   utilisation_rate, 
                                   aqi, 
                                   issue_time, 
                                   resident_count,
                                   industrial,
                                   mixed,
                                   residential]], 
                                 columns=feature_columns)
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return render_template('index.html', 
                             prediction_text=f'Estimated Electricity Cost: ₹ {prediction:.2f}')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)