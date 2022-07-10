from flask import *
import pandas as pd
import json, time
import joblib
import re
import os
from flask_cors import CORS


filename = 'Final_xgboost_model.joblib'
model = joblib.load(filename)
outlier_detector = joblib.load('KNN_outlier_detector.joblib')
precautions = pd.read_csv('disease_precautions.csv')
info = pd.read_csv('28_disease_info_translated.csv')


app = Flask(__name__)
CORS(app)

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
    if  sum(symptoms)<3:
        data_set = {'warning' : 'For a better result, please choose at least 3 symptoms'}
    else:
        out = outlier_detector.predict([symptoms])[0]
        if out == 1:
            data_set = {'warning' : 'I can help you to see a real doctor for a better result since your symptoms are away from my knowledge domain'}
        else:
            disease = model.predict([symptoms])[0]

            data_set = {'prediction': disease,
                        "precaution_1": precautions[precautions.Disease == disease].iloc[0][2],
                        "precaution_2": precautions[precautions.Disease == disease].iloc[0][3],
                        "precaution_3": precautions[precautions.Disease == disease].iloc[0][4],
                        "precaution_4": precautions[precautions.Disease == disease].iloc[0][5],
                        "prediction_in_arabic": precautions[precautions.Disease == disease].iloc[0][6],
                        "precaution_1_in_arabic": precautions[precautions.Disease == disease].iloc[0][7],
                        "precaution_2_in_arabic": precautions[precautions.Disease == disease].iloc[0][8],
                        "precaution_3_in_arabic": precautions[precautions.Disease == disease].iloc[0][9],
                        "precaution_4_in_arabic": precautions[precautions.Disease == disease].iloc[0][10],
                        
                        'Overview': info[info.disease == disease]['Overview'].values.tolist(),
                        'Causes': info[info.disease == disease]['Causes'].values.tolist(),
                        'Risk_Factors': info[info.disease == disease]['Risk factors'].values.tolist(),
                       
                        'Overview_in_arabic': info[info.disease == disease]['Overview_in_arabic'].values.tolist(),
                        'Causes_in_arabic': info[info.disease == disease]['Causes_in_arabic'].values.tolist(),
                        'Risk_Factors_in_arabic': info[info.disease == disease]['Risk_factors_in_arabic'].values.tolist()}
                     
    json_dump = json.dumps(data_set)
    return json_dump

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
