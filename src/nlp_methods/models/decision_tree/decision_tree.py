# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier


# Importing the dataset
dataset = pd.read_csv(r'C:\Users\MARINOS\Documents\zagkotsis\src\nlp_methods\models\decision_tree\nlp_data.csv')
X = dataset.iloc[:, 1:95].values
y = dataset.iloc[:, 95].values
#
# sc = StandardScaler()
# X = sc.fit_transform(X)
X = normalize(X)
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=7, shuffle=True)


# Fitting classifier to the Training set
# Create your classifier here
dtree_model = DecisionTreeClassifier(max_depth=3,criterion='gini').fit(X_train, y_train)

# Predicting the Test set results
y_pred = dtree_model.predict(X_test)

# Making the Confusion Matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test Accuracy
print("Test Accuracy  :: ", accuracy_score(y_test, y_pred))