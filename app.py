import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model4.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if(prediction[0]==0):

        output="I'm sorry , we cannot provide a loan to you "
        pic='https://i.pinimg.com/originals/23/b8/aa/23b8aa074e2b28f26d1fd3815872e5de.png'
                
    else:
        output="Your loan request has been accepted!"
        pic='https://i.pinimg.com/originals/6c/67/40/6c6740b3ddad811f0d920a85e4d8c222.png'
    return render_template('final.html', prediction_text='{}'.format(output),pic=pic)


if __name__ == "__main__":
    app.run(debug=True)