import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

# Base directory for absolute path resolution
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, 'Static')
template_dir = os.path.join(base_dir, 'templates')

app = Flask(
    __name__,
    static_folder=static_dir,
    static_url_path='/static',
    template_folder=template_dir
)

# Load scaler and trained RandomForestClassifier
scaler_path = os.path.join(base_dir, 'models', 'stroke-prediction-standard-scaler-10.pkl')
model_path = os.path.join(base_dir, 'models', 'stroke-prediction-10.pkl')

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

with open(model_path, 'rb') as file:
    rfc = pickle.load(file)

classes = ['No stroke', 'stroke']

feature_names = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]


def predict_stroke(gender, age, hypertension, heart_disease, ever_married,
                   work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    """
    Transforms raw input features and predicts stroke risk using the pre-trained model.
    Returns: (label, stroke_probability_percentage)
    """
    sample = pd.DataFrame(
        [[gender, age, hypertension, heart_disease, ever_married,
          work_type, Residence_type, avg_glucose_level, bmi, smoking_status]],
        columns=feature_names
    )
    sample_scaled = scaler.transform(sample)
    # Pass as numpy ndarray to match how RandomForest was trained and avoid sklearn feature name warnings
    prediction = rfc.predict(sample_scaled)
    probabilities = rfc.predict_proba(sample_scaled)[0]
    stroke_prob = round(float(probabilities[1]) * 100, 2)
    label = classes[prediction[0]]
    return label, stroke_prob


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = float(request.form['gender'])
        age = float(request.form['age'])
        hypertension = float(request.form['hypertension'])
        heart_disease = float(request.form['heart_disease'])
        ever_married = float(request.form['ever_married'])
        work_type = float(request.form['work_type'])
        Residence_type = float(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = float(request.form['smoking_status'])

        label, stroke_prob = predict_stroke(
            gender, age, hypertension, heart_disease, ever_married,
            work_type, Residence_type, avg_glucose_level, bmi, smoking_status
        )

        print(f'Prediction: {label} (Probability: {stroke_prob:.2f}%)')

        if stroke_prob >= 50.0:
            risk_level = 'High'
        elif stroke_prob >= 20.0:
            risk_level = 'Moderate'
        else:
            risk_level = 'Low'

        return render_template(
            'index.html',
            prediction=label,
            probability=stroke_prob,
            risk_level=risk_level,
            form_data=request.form
        )
    except (KeyError, ValueError) as e:
        error_msg = f'Invalid input: Please ensure all fields are filled with valid numerical values. ({e})'
        return render_template('index.html', error=error_msg, form_data=request.form), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
