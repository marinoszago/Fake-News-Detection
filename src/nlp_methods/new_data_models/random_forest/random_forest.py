# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\MARINOS\Documents\zagkotsis\src\nlp_methods\models\random_forest\nlp_data.csv')
X = dataset.iloc[:, 1:95].values
y = dataset.iloc[:, 95].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators = 20, criterion = 'gini'))
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Test Accuracy
print("Test Accuracy  :: ", accuracy_score(y_test, y_pred))
