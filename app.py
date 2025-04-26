import pickle
from flask import Flask,render_template,app,url_for,jsonify,request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app=Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaling=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaling.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
if __name__=='__main__':
    app.run(debug=True)

