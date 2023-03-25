Model Comparison - Flask Application
This Flask application is designed to accept user data and target variable from the user, preprocesses the data and run several different regression models on the data to find the best one based on accuracy, precision, recall and F1-score. Below are the different steps and their explanations in this application.

Getting Started
Prerequisites
This Flask application requires the following libraries:

Flask
Pandas
NumPy
Scikit-learn
XGBoost
To install these libraries, run the following command:

Copy code
pip install flask pandas numpy scikit-learn xgboost
Starting the Application
To start the Flask application, run the following command in the terminal:

Copy code
python app.py
This will start the application and the application can be accessed at http://localhost:5000.

Running the Application
Uploading the Data
To use the application, the user needs to upload a CSV file containing the data. The user can do this by clicking the "Choose File" button on the homepage of the application.

Selecting the Target Variable
After uploading the data, the user needs to specify the name of the target variable in the dataset. The target variable is the variable that the models will predict. The user can do this by entering the name of the target variable in the "Target Variable" field on the homepage of the application.

Running the Models
After uploading the data and selecting the target variable, the user can run the models by clicking the "Run Models" button on the homepage of the application. The application will preprocess the data, split the data into training and test sets, and run the following regression models on the training set:

Logistic Regression
Random Forest Classifier
Gradient Boosting Classifier
XGBoost Classifier
Support Vector Classifier
After running the models, the application will print the accuracy, precision, recall, and F1-score of each model on the test set. The application will also print the best model based on the accuracy score.

Code Explanation
The Flask application consists of two routes:

"/" route - This route returns the homepage of the application.
"/predict" route - This route accepts the data and target variable from the user, preprocesses the data, and runs the regression models on the data.
The "/predict" route contains the following steps:

Load the dataset from a CSV file
Drop variables that have more than 50 unique labels
Impute missing values
One-hot encode categorical variables
Scale numerical variables
Split the data into training and test sets
Run the regression models on the training set
Print the accuracy, precision, recall, and F1-score of each model on the test set
Print the best model based on the accuracy score
The regression models are defined using scikit-learn and XGBoost libraries. The accuracy, precision, recall, and F1-score of each model are calculated using scikit-learn metrics library.
