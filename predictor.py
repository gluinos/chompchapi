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

import os
import pickle
import sys
import time
import json
import requests
import random
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils import clean_data

from reduction import num_metacategories, metacategories

def get_yelp_data(latitude, longitude):
    url = "https://api.yelp.com/v3/businesses/search?limit=50&latitude={}&longitude={}".format(latitude,longitude)
    # return json.load(open("yelp/yelp_goleta.json"))
    # not safe, but .... sue me.
    r = requests.get(url, headers={"Authorization":"Bearer ZejdJl8XfQJ2JGQ2uRwoYDzeRqjRgcY_0tdBEsWM_Ey1QDaQLOeKu53K_vIS5Ks7G0Plt3R4UtD6P8JrUZ80CO3hu4y1pC8rFkw3T4wQmo57GCpYRW2IcMCUHW85XHYx"})
    return r.json()

def get_top_store_info(predvec, latitude, longitude):
    data = get_yelp_data(latitude, longitude)
    if len(data["businesses"]) == 0:
        raise Exception("No places found")
    stores = data["businesses"]
    score_stores = []
    for store in stores:
        categories = [x["title"] for x in store["categories"]]
        imcs = []
        score = 0.
        for cat in categories:
            vals = [imc*predvec[imc] for imc,mc in enumerate(metacategories) if cat in mc]
            score += sum(vals)
        score /= len(categories)
        if score == 0.: score += random.random()*0.3
        else: score += random.random()*0.3-0.15
        score_stores.append([score,store])
    score_stores = sorted(score_stores, reverse=True)
    return score_stores[0][1]


class Predictor(object):

    def __init__(self):

        self.max_len = 10
        self.max_words = 1500
        self.model = None
        self.X = None
        self.Y = None
        self.tokenizer = None

    def make_model(self):
        inputs = Input(name='inputs',shape=[self.max_len])
        layer = Embedding(self.max_words,50,input_length=self.max_len)(inputs)
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
        self.model = model

    def load_model(self):
        print("Loading model...")
        self.model = load_model('model.h5')
        print("Done loading model")

    def load_data(self):
        print("Loading data...")
        # also fits the tokenizer for sequence conversion later
        df = pd.read_pickle("combined_data.pkl")
        target_vecs = np.vstack([
            df["cat_{}".format(i)] for i in range(num_metacategories)
            ]).T
        if not self.tokenizer:
            print("Making tokenizer...")
            self.X = df.text
            self.Y = target_vecs
            tok = Tokenizer(num_words=self.max_words)
            tok.fit_on_texts(self.X)
            self.tokenizer = tok
            # print(sorted(self.tokenizer.word_counts.items(),key=lambda x:-x[1])[:self.max_words])
        print("Done loading")

    def train(self):
        print("Starting training...")
        X_train,X_test,Y_train,Y_test = train_test_split(self.X,self.Y,test_size=0.40)
        self.tokenizer.fit_on_texts(X_train)
        sequences = self.tokenizer.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences,maxlen=self.max_len)
        self.model.fit(
                sequences_matrix,Y_train,
                batch_size=512,epochs=25,
                validation_split=0.2,
                callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]
                )
        print("Saved model")
        self.model.save('model.h5')

    def predict_raw(self, thestring):
        """
        Return raw vector from model prediction
        """
        t0 = time.time()
        statement_arr = np.array([" ".join(clean_data(thestring))])
        test_sequences = self.tokenizer.texts_to_sequences(statement_arr)
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=self.max_len)
        predvec = self.model.predict(test_sequences_matrix)[0]
        t1 = time.time()
        print("Got prediction for {} in {:.2}secs".format(statement_arr,t1-t0))
        return predvec

    def predict(self, thestring):
        """
        Return parsed from model prediction
        """
        return self.predict_raw(thestring)


if __name__ == "__main__":

    predictor = Predictor()
    predictor.load_data()
    # predictor.load_model()

    predictor.make_model()
    predictor.train()

    # print(predictor.predict_raw("acidic cheap service good bad food"))
    # predvec = predictor.predict("")
    # print(get_top_store_info(predvec, 34.41, -119.85))

