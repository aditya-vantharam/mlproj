from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_csv('diabetes.csv')
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = [float(request.form[key]) for key in request.form]
        input_data = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
