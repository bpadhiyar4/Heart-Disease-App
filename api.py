from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)
ml_model = joblib.load('ml_model.joblib')

@app.route('/')
@app.route('/predict', methods=['POST'])
def Post():
    try:
        data = request.get_json()
        data = pd.DataFrame(data, index=[0])
        # print(data)
        response = make_response()
        # data = [57, 0, 0, 2, 120, 163, 0]
        prediction = ml_model.predict(np.array(data).reshape((1,-1)))
        response = jsonify({
            "statusCode": 200,
            "status": "Prediction Made",
            "result": str(prediction[0])
        })
        # response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as error:
        return jsonify({
            "statusCode": 500,
            "status": "Could not make prediction",
            "error": str(error)
        })

if __name__ == "__main__":
    app.run(host='127.0.0.1',port= 5000, debug=True)
