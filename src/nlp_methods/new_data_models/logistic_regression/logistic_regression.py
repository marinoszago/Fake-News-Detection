# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier






# Importing the dataset
dataset = pd.read_csv('../../../../data/nlp_data_with_annotations/new_nlp_data.csv')
X = dataset.iloc[:, 0:94].values
y = dataset.iloc[:, 94].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set

# classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial'))
classifier = LogisticRegression(solver='newton-cg', multi_class='multinomial')
classifier.fit(X_train, y_train)


# # Predicting the Test set results
y_pred = classifier.predict(X_test)
#
# # Making the Confusion Matrix
#
print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, classifier.predict(X_test)))
print(classifier.score(X_test, y_test))
print(np.mean(y_test == classifier.predict(X_test)))

# Test Accuracy

print("Test Accuracy  :: ", accuracy_score(y_test, y_pred))
