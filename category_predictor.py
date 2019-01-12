import os
import json
import pandas as pd

import tensorflow as tf
import numpy as np
import matplotlib as plt

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def generate_sequences(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    vocab_size = len(tokenizer.word_index) + 1
    print("vocab_size: ", vocab_size)
    x = tokenizer.texts_to_sequences(data)
    return x


# def one_hot_encode_labels(labels):
#     return labels


	
def preprocess_data(input_data):
    X_train, X_test, y_train, y_test = train_test_split(input_data[0], input_data[1], test_size=0.25, random_state=1000)
    
    X_train = generate_sequences(X_train)
    X_test = generate_sequences(X_test)
    X_train = pad_sequences(X_train, padding='post', maxlen=max_sequence_length)
    X_train = pad_sequences(X_test, padding='post', maxlen=max_sequence_length)

    return X_train, X_test, y_train, y_test




def create_model():
    model = Sequential()
    # model.add(Embedding(vocab_size, 50(Specift how many dimensions to represent word), input_length=seq_length))
    # keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', input_length=None)
    model.add(Embedding(vocab_size, vector_len, input_length=max_sequence_length))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.25)

    model.add(LSTM(hidden_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.25)

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_categories, activation='softmax'))

    return model


if __name__ == "__main__":

    data_path = ''
    epochs = 10
    batch_size = 16
    # vocab_size = 10000  #The size of vocabulary list being fed to network
    hidden_size = 100   # The number of hidden neurons for our LSTM layer
    vector_len = 50
    max_sequence_length = 500
    num_categories = 6


    x_train, x_test, y_train, y_test = preprocess_data(data_path)

    model = create_model()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']) 

    history = model.fit(x_train, y_train,
                    epochs=epochs,
                    verbose=False,
                    validation_data=(x_test, y_test),
                    batch_size=batch_size
                    )

    loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    # plot_history(history)
 
    # model.fit_generator()
    model.history()

    print(model.summary())



