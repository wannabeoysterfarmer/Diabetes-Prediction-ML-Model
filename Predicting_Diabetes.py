# Importing dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data collection and analysis using Pima Indians Diabetes dataset
# Loading data set:
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Print the first five rows of the dataset:
print(diabetes_dataset.head())

# Get number of rows and columns:
diabetes_dataset.shape

# Getting statistical measures of the data (columns):
diabetes_dataset.describe()

# Get counts of the different values for one feature:
diabetes_dataset['Outcome'].value_counts()

# Group by Outcome values, and then find the mean for other features:
diabetes_dataset.groupby('Outcome').mean()

# Separate the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Data standardization - turns it into standard distribution
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

# Train test split:
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')

# Training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# Model Evaluation - find the accuracy score
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) # compare prediction with original.

# Accuracy score (>75% is good)
print("Training data accuracy score:", training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of test data", test_data_accuracy)

# Building prediction system:
input_data = (4,110,92,0,0,37.6,0.191,30) # medical info (x, y, z, etc.)

# Change the input data to numpy array
input_data_as_numpy_array = np.asanyarray(input_data)

# Reshape the array as we are predicting for one instance only.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data 
std_data = scaler.transform(input_data_reshaped)
print("standardized data:", std_data)

prediction = classifier.predict(std_data)
print("prediction:", prediction) # predict whether person has diabetes or not. 

# Print conclusion based on results. 
if (prediction[0] == 0):
  print('The person is not diabetic.')
else:
  print('The person is diabetic.')

  