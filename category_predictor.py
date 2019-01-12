import json

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


# # truncate and pad input sequences


# X_train = tokenizer.texts_to_sequences(sentences_train)
# X_test = tokenizer.texts_to_sequences(sentences_test)

# max_words_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, padding=post, maxlen=max_words_length)



def generate_sequences(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    vocab_size = len(tokenizer.word_index) + 1
    print("vocab_size: ", vocab_size)
    
	
def preprocess_data(input_data):
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=1000)

    #Need to turn our words into vectors
    return x, y




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
    data_dir = ''
    epochs = 1
    batch_size = 1
    # vocab_size = 10000  #The size of vocabulary list being fed to network
    hidden_size = 100   # The number of hidden neurons for our LSTM layer
    vector_len = 50
    max_sequence_length = 500
    num_categories = len()




    model1 = create_model()
    # train_x, train_y = preprocess_data(data_path)
    # test_x, test_y = preprocess_data(data_path)

    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy']) 
    # model.fit_generator()
    model.fit(X, y, )
    model1.history()
    print(model1.summary())



