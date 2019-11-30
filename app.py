from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

my_random_forest=pickle.load(open("heart_disease.pkl","rb"))
app = Flask(__name__)

@app.route('/')
def home():
        return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
        #if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

             
                
        #vect = cv.transform(data).toarray()
        vect=np.array(data)
        vect=vect.astype(np.float64)
                #my_prediction = clf.predict(vect)

                #data=request.get_json(force=True)
                #predict_request=[data['chol']]
                #predict_request=[data['age'],data['sex'],data['cp'],data['trestbps'],data['chol'],data['fbs'],data['restecg'],data['thalach'],data['exang'],data['oldpeak'],data['slope'],data['ca'],data['thal']]
                #predict_request=np.array(predict_request)
                #predict_request=predict_request.reshape(1,-1)
                #print(vect)
        vect=vect.reshape(1,-1)
        print(vect)
        y_hat=my_random_forest.predict(vect)
        output=[y_hat[0]]
        print(output[0])
        return render_template('result.html',prediction = output[0])



if __name__ == '__main__':
        app.run(debug=True)
