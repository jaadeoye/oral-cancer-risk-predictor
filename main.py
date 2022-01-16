from flask import Flask, render_template, url_for, request
import pandas as pd 
import numpy as np 
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def analyze():
    if request.method == 'POST':
        V1 = request.form['V1']
        V3 = request.form['V3']
        V4 = request.form['V4']
        V6 = request.form['V6']
        V8 = request.form['V8']
        V9 = request.form['V9']
        V10 = request.form['V10']
        V11 = request.form['V11']
        V12 = request.form['V12']
        V13 = request.form['V13']
        V14 = request.form['V14']
        V15 = request.form['V15']
        V16 = request.form['V16']
        V17 = request.form['V17']
        V18 = request.form['V18']
        V20 = request.form['V20']
        V21 = request.form['V21']
        V22 = request.form['V22']
        V23 = request.form['V23']
        V25 = request.form['V25']
        V27 = request.form['V27']
        V28 = request.form['V28']
        V29 = request.form['V29']
        V30 = request.form['V30']
        V31 = request.form['V31']
        V33 = request.form['V33']
        V34 = request.form['V34']
        V35 = request.form['V35']
        
        model_choice = request.form['model_choice']

        sample_data = [V1, V3, V4, V6, V8, V9, V10, V11, V12, V13, V14,
       V15, V16, V17, V18, V20, V21, V22, V23, V25, V27,
       V28, V29, V30, V31, V33, V34, V35]

        ex1=np.array(sample_data).reshape(1,-1)

        if model_choice == 'AdaBoost':
            ada = pickle.load(open('ada', 'rb'))
            result_prediction = ada.predict(ex1)
            prob_prediction = ada.predict_proba(ex1)
        elif model_choice == 'KNeighborsClassifier':
            knn = pickle.load(open('knn', 'rb'))
            result_prediction = knn.predict(ex1)
            prob_prediction = knn.predict_proba(ex1)
        elif model_choice == 'DecisionTreeClassifier':
            dt = pickle.load(open('dt2', 'rb'))
            result_prediction = dt.predict(ex1)
            prob_prediction = dt.predict_proba(ex1)
            
    return render_template('predict.html', result_prediction = result_prediction, prob_prediction = prob_prediction, model_selected=model_choice)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=9050)
