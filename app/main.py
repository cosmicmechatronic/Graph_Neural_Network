from flask import Flask,jsonify, request
from torch_utils import transform_smile, get_prediction
import json
import requests


app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    #data = request.json
    errors = []
    if request.method == "POST":
        try:
            data = request.json
            string =json.dumps(data)
            value =string[11:len(string)-2]

            score=get_prediction(value)
            if score == 1:
                result = 'Toxic'
            if score == 0:
                result = 'Non-Toxic'

        except:
            errors.append(
                "Unable to get SMILE string. Please make sure it's valid and try again."
            )
            result = errors[0]


    response_dict = {"answear": result}
    return jsonify(response_dict)
