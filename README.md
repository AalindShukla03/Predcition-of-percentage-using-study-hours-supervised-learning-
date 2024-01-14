# Prediction-of-percentage-using-study-hours-supervised-learning-
This project seeks to explore the correlation between students' study hours and their respective exam scores. Utilizing linear regression, we construct a predictive model designed to estimate a student's exam performance, taking into account the duration dedicated to studying.
# Exam Score Prediction Model

## Objective:

The primary goal of this project is to develop a predictive model that estimates a student's exam score based on the number of hours they studied. The objective is to create an accurate machine learning model that can provide insights into the relationship between study hours and exam performance.

## About the Project:

In this Python project, we focus on predicting exam scores using a linear regression model. The dataset includes information about students' study hours and their corresponding exam scores.

## Dataset:

The dataset is collected from a source and includes two columns: 'Hours' representing the number of hours a student studied, and 'Scores' representing the exam scores achieved by the students.

## Additional Python Libraries Required:

- pandas
- matplotlib
- scikit-learn


## Step by Step Implementation of Linear Regression:

<ul>
  <li> 
    
# Importing necessary libraries
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt  

# Loading the dataset from a URL
    url = "http://bit.ly/w-data"
    s_data = pd.read_csv(url)
    print("Data imported successfully")

# Plotting the distribution of scores
    s_data.plot(x='Hours', y='Scores', style='o')  
    plt.title('Hours vs Percentage')  
    plt.xlabel('Hours Studied')  
    plt.ylabel('Percentage Score')  
    plt.show()

# Splitting the data into training and testing sets
    X = s_data.iloc[:, :-1].values  
    y = s_data.iloc[:, 1].values  
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Creating and training a linear regression model
    from sklearn.linear_model import LinearRegression  
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train) 
    print("Training complete.")
        
# Testing the model on the test set
    print(X_test)
    y_pred = regressor.predict(X_test)

# Predicting for a custom input
    hours = 9.25
    own_pred = regressor.predict(np.array([hours]).reshape(-1, 1))
    print("No of Hours = {}".format(hours))
    print("Predicted Score = {}".format(own_pred[0]))

# Evaluating the model using Mean Absolute Error
    from sklearn import metrics  
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

# Comparing Actual vs Predicted
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
    df 
</li>

## Step to integrate a flask app for real time implementation
## Creating a flask app
<ul>
  <li>
    
## Flask app code
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
        
 ## USER INTERFACE HTML
 You'll need to create two HTML templates, index.html and result.html, and place them in a folder named templates in the same directory as your Flask app.
 ## Rndex.html
     <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Study Hours Predictor</title>
    </head>
    <body>
        <h1>Enter your study hours</h1>
        <form action="/predict" method="post">
            <label for="hours">Study Hours:</label>
            <input type="text" id="hours" name="hours" required>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
 ## Result.html
     <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Result</title>
    </head>
    <body>
        <h1>Your Study Result Prediction</h1>
        <p>You have studied {{ hours }} hours.</p>
        <p>Predicted Score: {{ prediction }}</p>
        <a href="/">Go back to enter more hours</a>
    </body>
    </html>



</li>
