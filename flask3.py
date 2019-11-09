# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:19:09 2019

@author: ASUS
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
import smtplib

#import os
#os.getcwd()
app = Flask(__name__)
model = pickle.load(open('credit_card_fraud.pkl', 'rb'))
s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login("sender'smail@gmail", "password")

@app.route('/')
def home():
    return render_template('UI.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    if  output == 1:
        return render_template('UI.html', prediction ='This is a fraud transaction')
        message = "This is a fraud transaction"
        s.sendmail("sender'smail@gmail", "reciever'smail999@gmail", message)
        s.quit()
        
    elif output == 0:
        return render_template('UI.html', prediction ='This is a non fraud transaction')
        message = "This is not a fraud transaction"
        s.sendmail("sender'smail@gmail", "reciever'smail999@gmail", message)
        s.quit()


    

if __name__ == "__main__":
    app.run(debug=True)