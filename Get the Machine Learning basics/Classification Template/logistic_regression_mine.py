# -----Data Preprocessing

# -----Library imports (most common for machine learning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----Import the dataset
# Save data into variable using pandas
dataset = pd.read_csv('Social_Network_Ads.csv')
# The dependent variable is the one to be predicted
# The independent variables are the predictors

# We split the dependent and independent variables into X and Y
X = dataset.iloc[:, [2, 3]].values # Predictors 
Y = dataset.iloc[:, 4].values # What we are trying to predict

# -----Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split # Splitting library
# All the arrays are initialized at the same time
# Normally the test_size is a small percentage
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0) # 25% for test set

# -----Feature scaling (Important to make all variables equally important)
from sklearn.preprocessing import StandardScaler
# Instantiate the scaler for X
sc_X = StandardScaler()
# Apply the scaler for the X sets
X_train = sc_X.fit_transform(X_train)
# No need to fit the Scaler, it has already been done for X_train
X_test = sc_X.transform(X_test)

# We dont need to apply scaling for Y in this case, because there is only 0 or 1 (its normalized)

# -----Fitting Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression
# Create classifier with the Logistic Regression Object
classifier = LogisticRegression(random_state = 0)
# Fit to training sets X and Y
classifier.fit(X_train, Y_train)

# -----Predicting the test set results
Y_pred = classifier.predict(X_test) # Predictions based on the test set

# -----Making the confusion matrix (evaluate if the model understood the evaluation)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred) #Import real and predicted results

# -----Visualising the training set results

