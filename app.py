from flask import *
import pandas as pd
import json, time
import joblib
import re



filename = 'Final_xgboost_model.joblib'
model = joblib.load(filename)
outlier_detector = joblib.load('Angle-based.joblib')
precautions = pd.read_csv('disease_precautions.csv')


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home_page():
    data_set = {'Page': 'Home', 'Message': "Let's get started and send me your symptoms", 'Timestamp': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump

@app.route('/predict/', methods=['GET'])
def request_page():
    symptoms = request.args.get('symptoms') # /predict/?symptoms=symptoms
    symptoms= re.findall('\d', symptoms)
    symptoms= list(map(int,symptoms))

    out = outlier_detector.predict([symptoms])[0]
    if out == 1:
        data_set = {'message' : 'I can help you to see a real doctor for better result since your symptoms are away of my knowledge domain'}
    else:
        disease = model.predict([symptoms])[0]

        data_set = {'prediction': disease,
                    "precaution_1": precautions[precautions.Disease == disease].iloc[0][2],
                    "precaution_2": precautions[precautions.Disease == disease].iloc[0][3],
                    "precaution_3": precautions[precautions.Disease == disease].iloc[0][4],
                    "precaution_4": precautions[precautions.Disease == disease].iloc[0][5],
                            #results in Arabic
                        'disease': precautions[precautions.Disease == disease].iloc[0][6],
                        "precaution_5": precautions[precautions.Disease == disease].iloc[0][7],
                        "precaution_6": precautions[precautions.Disease == disease].iloc[0][8],
                        "precaution_7": precautions[precautions.Disease == disease].iloc[0][9],
                        "precaution_8": precautions[precautions.Disease == disease].iloc[0][10]}
    json_dump = json.dumps(data_set)
    return json_dump

if __name__=='__main__':
    app.run(port=3535)
