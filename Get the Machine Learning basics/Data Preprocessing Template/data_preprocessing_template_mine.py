# -----Data Preprocessing

# -----Library imports (most common for machine learning)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----Import the dataset
# Save data into variable using pandas
dataset = pd.read_csv('Data.csv')
# The dependent variable is the one to be predicted
# The independent variables are the predictors

# We split the dependent and independent variables into X and Y
X = dataset.iloc[:, :-1].values #Take all columns except the last one
Y = dataset.iloc[:, -1].values #Take only the last column
'''
# -----Taking care of missing data
from sklearn.preprocessing import Imputer
# Select what to target and which strategy to apply to deal with the problem
# Command + I to view help of class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Apply the imputer to x
# Only apply to columns with missing data (in this case colums 1 and 2)
imputer = imputer.fit(X[:, 1:3])
# Apply the transformation to the correct columns with assignation and the imputer
X[:, 1:3] = imputer.transform(X[:, 1:3])

# -----Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Encoding Class
labelencoder_X = LabelEncoder()
# Encode and replace the first column of the data
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # Take only the first column
# This class is used for dummy encoding (categorical data)
# These next 2 lines replace replaces each country by a column, in this case 3 countries = 3 columns
# In each column, 1 is marked for the appropriate country and the rest are 0 (for every entry)
# This is to make sure that the algorithm doesnt think any country is better than another
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# We use the normal label encoder for the Yes or No answers, as the weighting can be used for calculation
labelencoder_Y = LabelEncoder()
# Encode and replace the first column of the data
Y = labelencoder_Y.fit_transform(Y) # Take only the first column

'''
# The part in the block comment is not necessary. THe rest isa MUST and is ALWAYS template
# Copy and paste for preprocessing data

# -----Splitting the dataset into the Training Set and Test Set
from sklearn.model_selection import train_test_split # Splitting library
# All the arrays are initialized at the same time
# Normally the test_size is a small percentage
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) # 20% for test set

# -----Feature scaling (Important to make all variables equally important)
from sklearn.preprocessing import StandardScaler
# Instantiate the scaler for X
sc_X = StandardScaler()
# Apply the scaler for the X sets
X_train = sc_X.fit_transform(X_train)
# No need to fit the Scaler, it has already been done for X_train
X_test = sc_X.transform(X_test)

# We dont need to apply scaling for Y in this case, because there is only 0 or 1 (its normalized)
