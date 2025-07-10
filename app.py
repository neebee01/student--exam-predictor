from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('logistic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        study_hours = float(request.form['study_hours'])
        exam_score = float(request.form['exam_score'])

        # Prepare input for prediction
        input_data = np.array([[study_hours, exam_score]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Map output to label
        result = "Pass" if prediction == 1 else "Fail"

        return render_template('result.html', prediction=result)

    except:
        return render_template('result.html', prediction="Invalid input. Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)
