import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder,StandardScaler
import joblib
import pandas as pd
app = Flask(__name__)
model = joblib.load("train_model.pkl")
scaler = StandardScaler()

# fit the scaler on the training data
train_data = pd.read_csv('train dataset.csv')
le = LabelEncoder()
train_data['Gender'] = le.fit_transform(train_data['Gender'])
input_cols = ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality (Class label)']
train_data[input_cols] = scaler.fit_transform(train_data[input_cols])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      gender = request.form['gender']
      if(gender == "Female"):
        gender_no = 1
      else:
        gender_no = 2
      age = request.form['age']
      
      openness = request.form['openness']
      neuroticism = request.form['neuroticism']
      conscientiousness = request.form['conscientiousness']
      agreeableness = request.form['agreeableness']
      extraversion = request.form['extraversion']
      result = np.array([gender_no, age, openness,neuroticism, conscientiousness, agreeableness, extraversion], ndmin = 2)
      
      # use the same scaler to transform the test data
      final = scaler.transform(result)
      personality = str(model.predict(final)[0])
      return render_template("submit.html",gender = gender,age = age,openness = openness,neuroticism=neuroticism,conscientiousness=conscientiousness,agreeableness=agreeableness,extraversion=extraversion, answer = personality)

if __name__ == '__main__':
    app.run()
