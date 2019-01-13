import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping

import time
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils import clean_data

from reduction import num_metacategories

def get_model(max_words, max_len):
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(128)(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(256)(layer)
    layer = Dropout(0.5)(layer)
    layer = Activation('relu')(layer)
    layer = Dense(64)(layer)
    layer = Dropout(0.5)(layer)
    layer = Activation('relu')(layer)
    layer = Dense(num_metacategories)(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    model.summary()
    model.compile(loss='mse',optimizer=RMSprop(),metrics=['accuracy'])
    return model


if __name__ == "__main__":

    df = pd.read_pickle("combined_data.pkl")
    # df = df[:5000]
    # df["text"] = df.text.str.join(" ")

    # Unpack column by column into an num_review-by-num_metacategories matrix again
    target_vecs = np.vstack([
        df["cat_{}".format(i)] for i in range(num_metacategories)
        ]).T


    max_words = 2000
    max_len = 10
    X = df.text
    Y = target_vecs
    le = LabelEncoder()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.40)
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    print(sorted(tok.word_counts.items(),key=lambda x:-x[1])[:max_words])

    model = get_model(max_words,max_len)


    model.fit(
            sequences_matrix,Y_train,
            batch_size=512,epochs=15,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]
            )
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model.h5')

    # Now predict with a slightly insensitive sentence
    # NOTE: tok has to be fit on the X_train above or else the word vectors won't be correct
    model = load_model('my_model.h5')
    t0 = time.time()
    statement_arr = np.array([" ".join(clean_data("the chicken tikka masala at this indian restaurant was absolutely dope. I can't wait to go back again and try out the chicken korma while watching the waiter serve me with a turban and give me naan"))])
    test_sequences = tok.texts_to_sequences(statement_arr)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    print (statement_arr)
    print (model.predict(test_sequences_matrix))
    print time.time()-t0


