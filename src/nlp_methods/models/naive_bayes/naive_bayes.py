# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier




# Importing the dataset
dataset = pd.read_csv(r'C:\Users\MARINOS\Documents\zagkotsis\src\nlp_methods\models\naive_bayes\nlp_data.csv')
X = dataset.iloc[:, 1:95].values
y = dataset.iloc[:, 95].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling

# sc = StandardScaler()
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
# classifier = MultinomialNB()
classifier = OneVsRestClassifier(BernoulliNB())

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

print(confusion_matrix(y_test, y_pred))



# Test Accuracy
print("Test Accuracy  :: ", accuracy_score(y_test, y_pred))
