from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC

def LogisticRegressionModel(df,scaler):

    X = df.iloc[:, 0:27].values
    y = df.iloc[:, 27].values

    X = preprocessing.normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_reg = LogisticRegression(multi_class='multinomial',solver='lbfgs')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    score = log_reg.score(X_test, y_test)
    print(score)
    print(conf_mat)
    print(classification_report(y_test, y_pred, digits=3))
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_mat, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()

def NaiveBayesModel(df,scaler):

    X = df.iloc[:, 0:27].values
    y = df.iloc[:, 27].values

    X = preprocessing.normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    multi_nb = MultinomialNB()
    multi_nb.fit(X_train,y_train)

    y_pred = multi_nb.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    score = multi_nb.score(X_test, y_test)
    print(score)
    print(conf_mat)
    print(classification_report(y_test, y_pred, digits=3))
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_mat, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()

def RandomForestModel(df,scaler):
    X = df.iloc[:, 0:27].values
    y = df.iloc[:, 27].values

    X = preprocessing.normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    random_f = RandomForestClassifier(n_estimators = 80, criterion = 'entropy', random_state = 42)
    random_f.fit(X_train, y_train)

    y_pred = random_f.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    score = random_f.score(X_test, y_test)
    print(score)
    print(conf_mat)
    print(classification_report(y_test, y_pred, digits=3))
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_mat, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()

def SMVModel(df,scaler):
    X = df.iloc[:, 0:27].values
    y = df.iloc[:, 27].values

    X = preprocessing.normalize(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    conf_mat = confusion_matrix(y_test, y_pred)
    score = svm.score(X_test, y_test)
    print(score)
    print(conf_mat)
    print(classification_report(y_test, y_pred, digits=3))
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_mat, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()


if __name__=='__main__':
    df = pd.read_csv('F:\\zagkotsis\\data\\nlp_data_with_annotations\\new_nlp_data.csv')
    sc = StandardScaler()
    mm = preprocessing.MinMaxScaler()
    # LogisticRegressionModel(df,sc)
    # NaiveBayesModel(df,mm)
    # RandomForestModel(df,sc)
    SMVModel(df,sc)