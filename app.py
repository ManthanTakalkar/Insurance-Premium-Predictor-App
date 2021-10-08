import pickle
import os
import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import psycopg2
import pandas as pd

app = Flask(__name__)

con = psycopg2.connect(host="ec2-54-164-22-242.compute-1.amazonaws.com", port='5432', user="aybobrosixlybq", password="587d5f01c1d0490a914659179415d32551d1e297411998276c7a0a5e0422f5d1", database="dedq6ael2chsp0")
cur = con.cursor()

cur.execute('create table if not exists insurance(age int, gender varchar(15),bmi NUMERIC(5,2), children int,smoker varchar(15), region varchar(25), expenses NUMERIC)')
con.commit()


@cross_origin()
@app.route('/', methods=['GET'])
def homepage():
    return render_template('page1.html')


@cross_origin()
@app.route('/form', methods=['GET'])
def form():
    return render_template('form.html')

@cross_origin()
@app.route('/About', methods=['GET'])
def about():
    return render_template('AppInfo.html')

@cross_origin()
@app.route('/contact', methods=['GET'])
def developer():
    return render_template('Developer.html')


@cross_origin()
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    with open("./BestModel/RobustScaler.pkl", 'rb') as f:
        scalar = pickle.load(f)

    with open("./BestModel/FinalModel_ForPrediction.pkl", 'rb') as f:
        model = pickle.load(f)

    if request.method == 'POST':
        age = int(request.form['Age'])
        gender = (request.form["gender"])
        if gender == 'female':
            gender = 0
        else:
            gender = 1
        bmi = float(request.form['bmi'])
        children = int(request.form["children"])
        smoker = (request.form['smoker'])
        if smoker == 'yes':
            smoker = 1
        else:
            smoker = 0
        region = (request.form["region"])
        if region == 'northeast':
            region = 0, 0, 0
        elif region == 'northwest':
            region = 1, 0, 0
        elif region == 'southeast':
            region = 0, 1, 0
        else:
            region = 0, 0, 1

        d = [[age, bmi, children, gender, smoker, *region]]
        print(d)
        scaled_data = scalar.transform(d)
        print(scaled_data)
        prediction = model.predict(scaled_data)
        predict1 = prediction
        print(predict1)

        col1 = int(request.form['Age'])
        col2 = request.form["gender"]
        col3 = float(request.form['bmi'])
        col4 = int(request.form["children"])
        col5 = request.form['smoker']
        col6 = request.form['region']

        cur.execute(f'insert into insurance values{(col1, col2, col3, col4, col5, col6, round(predict1[0],2))}')
        con.commit()

        return render_template('output.html', result=f"Your Insurance Premium is: {round(predict1[0],2)}")

    else:
        return render_template('page1.html')


if __name__ == "__main__":
    app.run(debug=True)
    host = '0.0.0.0'
    port = 5432
