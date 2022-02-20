import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sys
import logging
app = Flask(__name__,template_folder='template')
model = pickle.load(open('model-bmi.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 0:
        output = "Extremely Weak - Health is the first priority, focus on your diet"
    elif prediction == 1:
        output = "Weak - Add some proteins and calories in your diet plan"
    elif prediction == 2:
        output = "Normal - cool! you are healthy"
    elif prediction == 3:
        output = "Overweight - reduce the intake of sugar and calories"
    elif prediction == 4:
        output = "Obese - burn the fat, work out regularly with diet plan"
    elif prediction == 5:
        output = "Extremely Obese - your health is your responsibility, consult a doctor"
    return render_template("index.html", prediction_text = output)
if __name__ == "__main__":
    app.run( debug=True)
