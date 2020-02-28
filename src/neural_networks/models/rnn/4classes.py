import numpy as np
import pandas as pd
import keras
from sklearn import preprocessing
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

# dataset = pd.read_csv('../../../../data/embeddings/word2vec/word2vec_with_stopwords 3 classes.csv')
# dataset = pd.read_csv('../../../../data/embeddings/word2vec/UNB_w2v.csv')
# seed = 7
# dataset = pd.read_csv('../../../../data/news_sample.csv')
# dataset = pd.read_csv('../../../../data/embeddings/word2vec/w2v_fake_news.csv')
dataset = pd.read_csv('../../../../data/embeddings/some_classes.csv',header=None)

# a = dataset.count()
df = dataset.drop_duplicates(keep=False)
# b = df.count()

X = df.iloc[:, 0:300].values
y = df.iloc[:, 300].values

y = np_utils.to_categorical(y, num_classes=4)
#
# X = preprocessing.normalize(X)
# sc = StandardScaler()
# X = sc.fit_transform(X)


sgd = optimizers.SGD(lr=0.01, decay=1e-6, nesterov=True)  # momentum=0.9
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


def build_ann(activation1, activation2, activation3, activation4, dropout, optimizer):
    classifier = Sequential()

    classifier.add(Dense(units=32, kernel_initializer='normal', activation=activation1, input_dim=300))
    # classifier.add(BatchNormalization())
    # classifier.add(Dropout(dropout))
    classifier.add(Dense(units=64, activation=activation2))
    # classifier.add(Dropout(dropout))
    classifier.add(Dense(units=64, activation=activation3))
    # classifier.add(Dropout(dropout))
    classifier.add(Dense(units=64, activation=activation4))
    # classifier.add(Dropout(dropout))
    # classifier.add(Dense(units=16, activation=activation3))
    classifier.add(Dense(units=4, activation=activation4))

    # Logo multiclass to logiko einai auto (30-35% accuracy)
    # classifier.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])

    # Logo omos tou one hot sto y epeidi einai se 0-1 isos paizei auto san lush(83% accuracy)
    # classifier.compile(optimizer=optimizer,
    #                    loss='binary_crossentropy',
    #                    metrics=['accuracy'])

    classifier.compile(optimizer=optimizer,
                       loss='categorical_crossentropy',
                       metrics=['acc',keras.metrics.categorical_accuracy])

    return classifier


def convert_y_test(y_test):
    temp = []
    for row in y_test:
        if row[0] == 1:
            temp.append(0)
        elif row[1] == 1:
            temp.append(1)
        elif row[2] == 1:
            temp.append(2)
        elif row[3] == 1:
            temp.append(3)
        elif row[4] == 1:
            temp.append(4)
        elif row[5] == 1:
            temp.append(5)
        elif row[6] == 1:
            temp.append(6)
        elif row[7] == 1:
            temp.append(7)
        elif row[8] == 1:
            temp.append(8)
        elif row[9] == 1:
            temp.append(9)
        elif row[10] == 1:
            temp.append(10)

    return temp


def plot_accuracy(history, version):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylim(0, 1)
    plt.title('model accuracy {}'.format(version))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_loss(history, version):
    # summarize history for loss
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.ylim(0, 1)
    plt.title('model loss {}'.format(version))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

results = []
for seed in [50]:
    dropout = 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    classifier = build_ann(activation1='relu',
                           activation2='relu',
                           activation3='relu',
                           activation4='softmax',
                           dropout=dropout,
                           optimizer=sgd)

    print("test")
    history = classifier.fit(X_train, y_train,
                             validation_split=0.2,
                             batch_size=32,
                             epochs=200,
                             shuffle=False,
                             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=10,
                                verbose=1,
                                mode='auto',
                                baseline=None)]
                                                      )

    # validation_split=0.2

    y_pred = classifier.predict(X_test)
    y_pred_classes = classifier.predict_classes(X_test)

    # convert y_test into array for feeding scikit learn methods
    y_check = convert_y_test(y_test)
    classifier.summary()

    print(classifier.layers)
    print(history)

    print('Confusion Matrix')
    print(confusion_matrix(y_true=y_check, y_pred=y_pred_classes))
    # print(confusion_matrix(y_true=y_test, y_pred=y_pred))
    print('Classification report')
    # print(classification_report(y_true=y_test, y_pred=y_pred))
    print(classification_report(y_true=y_check, y_pred=y_pred_classes))
    print(keras.metrics.categorical_accuracy(y_true=y_test, y_pred=y_pred))

    print("Validation Categorical Accuracy: " + str(np.average(history.history['val_acc'])))
    print("Test Categorical Accuracy: " + str(np.average(history.history['acc'])))
    print("Number of layers used: " + str(len(classifier.layers)))

    # Plot results
    version = "No Scaler all relu, sgd(custom) \n 3 layers, 75-25 & 90-10{}"
    plot_accuracy(history, version)
    plot_loss(history, version=version)
    results.append(np.average(history.history['acc']))


print(results)