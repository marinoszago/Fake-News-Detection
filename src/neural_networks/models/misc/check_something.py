# -*- coding: utf-8 -*-
"""
Created on Tue Nov  15 12:30:17 2018

@author: ggravanis
"""

import gzip
import os
import pickle

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer






# load the dataset
def load_dataset(fname='ms_cntk_atis.train.pkl.gz'):
    with gzip.open(os.path.join(data_dir, fname), 'rb') as stream:
        dataset, dicts = pickle.load(stream)
    print('Done  loading: ', fname)
    print('      samples: {:4d}'.format(len(dataset['query'])))
    print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
    print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
    print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return dataset, dicts


train_dataset, dicts = load_dataset('ms_cntk_atis.train.pkl.gz')
test_dataset, _ = load_dataset('ms_cntk_atis.test.pkl.gz')
word2idx, slot2idx, intent2idx = dicts['token_ids'], dicts['slot_ids'], dicts['intent_ids']

intent_tr, query_tr, slots_tr = train_dataset['intent_labels'], train_dataset['query'], train_dataset['slot_labels']
intent_te, query_te, slots_te = test_dataset['intent_labels'], test_dataset['query'], test_dataset['slot_labels']

# Create index to word/label dicts
idx2word = {word2idx[k]: k for k in word2idx}
idx2slot = {slot2idx[k]: k for k in slot2idx}
idx2intent = {intent2idx[k]: k for k in intent2idx}

# For conlleval script
words_train = [list(map(lambda x: idx2word[x], w)) for w in query_tr]
slots_train = [list(map(lambda x: idx2slot[x], w)) for w in slots_tr]
labels_train = [list(map(lambda x: idx2intent[x], y)) for y in intent_tr]

words_test = [list(map(lambda x: idx2word[x], w)) for w in query_te]
slots_test = [list(map(lambda x: idx2slot[x], w)) for w in slots_te]
labels_test = [list(map(lambda x: idx2intent[x], y)) for y in intent_te]

n_classes = len(idx2intent)  # number of intents
n_vocab = len(idx2word)  # number of words used
n_slots = len(idx2slot)  # number of entities

# Αυτά είναι εδώ γιατί....???? code refactored Keep only the first intent / label
y_train = []

for i in labels_train:
    if '+' in i[0]:
        y_train.append(i[0].split('+')[0])
    else:
        y_train.append(i[0])

y_test = []
for i in labels_test:
    if '+' in i[0]:
        y_test.append(i[0].split('+')[0])
    else:
        y_test.append(i[0])

df_train = pd.DataFrame(
    {
        'label': y_train,
        'query': words_train,
        'slot': slots_train
    }, columns=['query', 'slot', 'label'])
df_test = pd.DataFrame(
    {
        'label': y_test,
        'query': words_test,
        'slot': slots_test
    }, columns=['query', 'slot', 'label'])

# take the first intent for those with more than one intent
# df_train.label = df_train.label.apply(lambda x: x.split('+')[0] if '+' in x else x)
# df_test.label = df_test.label.apply(lambda x: x.split('+')[0] if '+' in x else x)

# y_train = df_train.label
# y_test = df_test.label

le = LabelBinarizer()

all_labels = set(y_train)
all_labels.update(y_test)

le.fit(list(all_labels))

Y = le.transform(y_train)
# Y = Y.reshape(-1,1)
Y_test = le.transform(y_test)
# Y_test = Y_test.reshape(-1,1)

X = [' '.join(sen) for sen in words_train]
X_test = [' '.join(sen) for sen in words_test]

'''Process the data
Tokenize the data and convert the text to sequences.
Add padding to ensure that all the sequences have the same shape.'''
max_words = 891
max_len = 35
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
# test data preprocess
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

# create embedding matrix from pretrained word2vec embeddings
embedding_matrix = np.zeros((max_words, 300))
for word, index in tok.word_index.items():
    if index > max_words - 1:
        break
    else:
        try:
            embedding_vector = model.wv[word]
            embedding_matrix[index] = embedding_vector
        except KeyError:
            continue



test_history = {}
iterations = 50
for i in range(1, iterations):
    temp_dict = {}

    model = Sequential()
    model.add(Embedding(max_words, 300, input_length=max_len, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(100, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(18, activation='softmax'))
    model.summary()  # prints a summary representation of the model.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(sequences_matrix, Y,
                        batch_size=64,
                        epochs=160,
                        validation_split=0.20,
                        verbose=2
                        # callbacks=[EarlyStopping(monitor='val_loss',
                        #                          min_delta=0,
                        #                          patience=10,
                        #                          verbose=1,
                        #                          mode='auto',
                        #                          baseline=None)])
                        )
    test_accuracy = model.evaluate(test_sequences_matrix, Y_test)

    print(
        'Test set\n  Iteration: {} \n Loss: {:0.3f}\n  Accuracy: {:0.6f}'.format(i, test_accuracy[0], test_accuracy[1]))

    temp_dict["loss"] = test_accuracy[0]
    temp_dict["accuracy"] = test_accuracy[1]
    test_history[i] = temp_dict

df = pd.DataFrame.from_dict(test_history)
df = df.transpose()
df.to_csv("politifact_Multiple_iterations.csv")

print(df)


def plots():
    # plot results and check for over - fitting
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    acc = history_dict["acc"]
    val_acc = history_dict["val_acc"]
    epochs = range(1, len(acc) + 1)

    # plots
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    # Evaluate the model on the test set
