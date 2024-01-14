from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the data
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)

# Train the linear regression model
X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values 
regressor = LinearRegression()  
regressor.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        hours = float(request.form['hours'])
        # Make prediction using the model
        prediction = regressor.predict([[hours]])[0]
        return render_template('result.html', hours=hours, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
