import numpy as np
import pandas as pd
from sklearn.svm import SVC

import keras
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, LSTM, Activation
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.utils import shuffle
from keras import optimizers
from sklearn.model_selection import KFold
from keras.utils import np_utils
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# dataset = pd.read_csv('../../../../data/embeddings/word2vec/word2vec.csv')
dataset = pd.read_csv('../../../../data/embeddings/allinone.csv')
seed = 7

X = dataset.iloc[:, 0:300].values
y = dataset.iloc[:, 300].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


# svm_params = {'SVM__kernel': ['linear'], 'SVM__C': [1, ], 'SVM__gamma': [0.1]}
svm_params = {'SVM__kernel': ['linear', 'rbf'], 'SVM__C': [1, 10, 100, 500, 1000], 'SVM__gamma': [0.1, 0.01, 0.001]}
pipelines = [('SVM', Pipeline([('Normalize', Normalizer()), ('SVM', SVC())]), svm_params)]

if __name__ == '__main__':
    seed = 7
    folds = 7

    for name, model, parameters in pipelines:
        grid_search = GridSearchCV(model, param_grid=parameters, cv=folds, verbose=1, return_train_score=True, n_jobs=7)

        print("Split sets")

        grid_search.fit(X_train, y_train)
        df = pd.DataFrame.from_dict(grid_search.cv_results_)
        df.to_csv('CrossValidation.csv')

        y_pred_test = grid_search.predict(X_test)
        y_pred_train = grid_search.predict(X_train)

        print(classification_report(y_pred=y_pred_test, y_true=y_test))
