from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data as JSON
        data = request.get_json()

        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])

        # Make a prediction
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def test():
    return "Server is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
