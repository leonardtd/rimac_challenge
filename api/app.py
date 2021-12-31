import flask
from flask import Flask, jsonify, request
import json
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)
def load_models():
    file_name = "../model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

def load_encoder():
    with open('../encoder', 'rb') as pickle_file:
        encoder = pickle.load(pickle_file)
    return encoder

@app.route('/predict', methods=['GET'])
def predict():
    # parse input features from request
    request_json = request.get_json()

    df = pd.DataFrame({"Age":[int(request_json['age'])],
                       "Sex":[request_json['sex']],
                       "ChestPainType":[request_json['chestPainType']],
                       "RestingBP":[int(request_json['restingBP'])],
                       "Cholesterol":[int(request_json['cholesterol'])],
                       "FastingBS":[int(request_json['fastingBS'])],
                       "RestingECG":[request_json['restingECG']],
                       "MaxHR":[int(request_json['maxHR'])],
                       "ExerciseAngina":[request_json['excerciseAngina']],
                       "Oldpeak":[float(request_json['oldpeak'])],
                       "ST_Slope":[request_json['sTSlope']],
                        })


    #Preparar la data con el mismo encoder usado en el entrenamiento
    s = (df.dtypes == 'object')
    object_cols = list(s[s].index)

    encoder = load_encoder()
    OH_encoder = encoder
    OH_cols_test = pd.DataFrame(OH_encoder.transform(df[object_cols]))

    OH_cols_test.index = df.index
    num_X_test = df.drop(object_cols, axis=1)
    oh_x_test = pd.concat([num_X_test, OH_cols_test], axis=1)
    
    #Cargar el modelo y generar prediccion
    model = load_models()
    preds = model.predict_proba(oh_x_test)[0]
    pred = preds[1] #1: La probabilidad de que sea positivo
    print(preds)
    response = json.dumps({'prob':str(round(pred, 2))})
    return response, 200

if __name__ == '__main__':
    app.run(debug=True)