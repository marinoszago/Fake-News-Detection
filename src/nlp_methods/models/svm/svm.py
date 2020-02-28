# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


# Importing the dataset
dataset = pd.read_csv(r'C:\Users\MARINOS\Documents\zagkotsis\src\nlp_methods\models\svm\nlp_data.csv')
X = dataset.iloc[:, 1:95].values
y = dataset.iloc[:, 95].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set

# classifier = SVC(kernel='sigmoid', decision_function_shape='ovr')
# classifier = SVC(kernel='sigmoid', decision_function_shape='ovo')
# classifier = SVC(kernel='rbf', decision_function_shape='ovo')
# classifier = SVC(kernel='rbf', decision_function_shape='ovr')
# classifier = SVC(kernel='poly', decision_function_shape='ovo')
# classifier = SVC(kernel='poly', decision_function_shape='ovr')
# classifier = SVC(kernel='linear', decision_function_shape='ovo')
# classifier = SVC(kernel='linear', decision_function_shape='ovr')
classifier = OneVsOneClassifier(SVC(gamma='auto'))
# classifier = OneVsRestClassifier(LinearSVC(class_weight="balanced"))
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

print(confusion_matrix(y_test, y_pred))

# Test Accuracy
print("Test Accuracy  :: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))