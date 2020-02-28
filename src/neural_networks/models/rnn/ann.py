import numpy as np
import pandas as pd
import keras
from sklearn import preprocessing
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split,KFold
from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization
import matplotlib.pyplot as plt
from keras import regularizers
from keras import optimizers
from keras.utils import np_utils

seed = np.random.seed()
dataset = pd.read_csv('C:/Users/MARINOS/Documents/zagkotsis/data/embeddings/word2vec/word2vec.csv')

X = dataset.iloc[:, 0:300].values
y = dataset.iloc[:, 300].values

y = np_utils.to_categorical(y, num_classes=6)

X = preprocessing.normalize(X)
# sc = StandardScaler()
# X = sc.fit_transform(X)


#Giwrgo allaxe to test size se 0.8 na deis auto poy sou elega alla bale kai categorical_crossentropy
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.25,train_size=0.75, shuffle=True, random_state=seed)
X_train, X_val, y_train,y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False,random_state=seed)


classifier = Sequential()

classifier.add(Dense(units=128,activation='relu', input_dim=300))
classifier.add(BatchNormalization()) #deal with overfit
classifier.add(Dropout(0.1)) #deal with overfit
classifier.add(Dense(units=64,activation='relu'))
classifier.add(Dropout(0.1)) #deal with overfit
classifier.add(Dense(units=32,activation='relu'))
classifier.add(Dense(units=6, activation='sigmoid'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.000007,beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#Logo multiclass to logiko einai auto (30-35% accuracy)
# classifier.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])

#Logo omos tou one hot sto y epeidi einai se 0-1 isos paizei auto san lush(83% accuracy)
classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['acc'])
history = classifier.fit(X_train, y_train,validation_data=(X_val, y_val), batch_size=32, epochs=100,shuffle=False)

y_pred = classifier.predict(X_test)
classifier.summary()

print(classifier.layers)
score, acc = classifier.evaluate(X_test, y_test, batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
print("Validation Accuracy: "+str(history.history['val_acc']))
print("Number of layers used: "+str(len(classifier.layers)))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()