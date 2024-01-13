from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)

# load the previously trained model
model = pickle.load(open("ad_prediction_model.pkl", "rb"))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    # get the user input from the HTML form
    a = float(request.form['a'])
    b = float(request.form['b'])
    c = float(request.form['c'])
    d = float(request.form['d'])
    e = int(request.form['e'])

    # create a numpy array with the user input
    input_data = np.array([[a, b, c, d, e]])

    # make a prediction using the model
    prediction = model.predict(input_data)[0]

    # display the prediction
    if prediction == 1:
        return render_template('index.html', prediction_text='User will click on Ad')
    else:
        return render_template('index.html', prediction_text='User will not click on Ad')

if __name__ == '__main__':
    application.run(debug=True)
