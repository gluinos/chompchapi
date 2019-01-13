import json
import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer




def load_json(path):
    # df = pd.DataFrame()
    df = pd.read_json(path)
    return df

def pd_to_json(df):
    json_output = df.to_json()
    return json_output


def tokenize_input(text_input):
    tokens = nltk.word_tokenize(text_input)
    return tokens

def clean_data(data):
    tokens = word_tokenize(data)
    # convert all words to lowercase
    tokens = [w.lower() for w in tokens]
    # strip punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # make sure word is alphabetical
    words = [word for word in stripped if word.isalpha()]
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # stem words to make them as their root word
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]

    return words

if __name__ == "__main__":

    print("Running program.")