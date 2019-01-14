import pandas as pd
import numpy as np
import string

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping

from tqdm import tqdm

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

from reduction import num_metacategories

def get_model(max_words, max_len):
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(128)(layer)
    layer = Dense(256)(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.50)(layer)
    layer = Dense(num_metacategories)(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    # model = Sequential(inputs=inputs,outputs=layer)

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def second_model(max_words, max_len):
    model = Sequential()
    # model.add(Embedding(vocab_size, 50(Specift how many dimensions to represent word), input_length=seq_length))
    # keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', input_length=None)
    model.add(Embedding(max_words, 50, input_length=max_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))

    model.add(LSTM(64, return_sequences=True)) 
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    model.add(LSTM(32))

    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_metacategories, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    

    return model

if __name__ == "__main__":

    df = pd.read_pickle("combined_data.pkl")
    df = df[:5000]

    # Unpack column by column into an num_review-by-num_metacategories matrix again
    target_vecs = np.vstack([
        df["cat_{}".format(i)] for i in range(num_metacategories)
        ]).T


    max_words = 1500
    max_len = 300
    epochs = 20
    learning_rate = 0.1
    epsilon = ''
    decay = ''

    X = df.text
    Y = target_vecs
    le = LabelEncoder()
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.40)
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    print(sorted(tok.word_counts.items(),key=lambda x:-x[1])[:max_words])

    # model = get_model(max_words,max_len)
    model = second_model(max_words,max_len)

    history = model.fit(
                sequences_matrix,Y_train,
                batch_size=64,epochs=epochs,
                validation_split=0.2,
                # callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]
                )

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    accr = model.evaluate(test_sequences_matrix,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    model.save('my_model_2.h5')  # creates a HDF5 file 'my_model.h5'



    # returns a compiled model
    # identical to the previous one
    model = load_model('my_model_2.h5')

    # Now predict with a slightly insensitive sentence
    # NOTE: tok has to be fit on the X_train above or else the word vectors won't be correct
    statement_arr = np.array([" ".join(clean_data("the chicken tikka masala at this indian restaurant was great"))])
    # model = load_model('my_model_two.h5')
    test_sequences = tok.texts_to_sequences(statement_arr)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

    print (statement_arr)
    print (model.predict(test_sequences_matrix))

    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.show(["plot"])



