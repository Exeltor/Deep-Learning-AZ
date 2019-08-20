#Artificial Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
# This is the classification template from the machine learning basics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
# This is from the machine learning basigs --- Categorical Data -----
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # Encoder for countries
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder() # Encoder for genders
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # Create dummy variables for countries
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # Remove dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # 2O% for test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN
# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential() # THe ANN itself

# Adding the input layer and the first hidden layer with dropout
'''
To calculate the dimension of the hidden layers
Take the average of dimension of the input layer and the output layer, in this case:
dim input layer = 11, dim output layer = 1 => (11+1)/2 = 6
relu = rectifier function
input_dim is necessary in the first layer as this is the first hidden layer
'''
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) # Dense is the layer object
classifier.add(Dropout(rate = 0.1)) # Adds dropout to first hidden layer

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) # Dense is the layer object
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
# Activation function is sigmoid because the result is binary
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) # Dense is the layer object

# Compiling the ANN
# 'adam' is a very efficient stockastic gradient descent algorithm
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # Change results to booleans on the 0.5 threshold

'''
Predicting a single new observation
Predict if the customer with the following information will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
'''
# We have to input the formatted variables (like the ones from preprocessing)
# use the scaler 'sc' to scale the data appropriately
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5) # Change results to booleans on the 0.5 threshold

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def buildClassifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = buildClassifier, batch_size = 10, epochs = 100)
# cv = 10 recommended most of the time
# n_jobs = -1 to use all cpus
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def buildClassifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = buildClassifier)
# Dictionary to hold the testing parameters
parameters = {
    'batch_size' : [25, 32],
    'epochs' : [100, 500],
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
# Vars to hold the best results
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_




