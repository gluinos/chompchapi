import json
import nltk
import pandas as pd
import string
import nltk
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
    """
    Takes string
    Returns list of words after doing the following
    - stripping out nonalpha
    - tokenizing
    - lowercasing
    - removing stopwords
    - stemming
    - getting only the top 10 most common words
    """
    tokens = word_tokenize(data)
    # convert all words to lowercase
    tokens = [w.lower() for w in tokens]
    # make sure word is only alphabetical
    words = [word for word in tokens if word.isalpha()]
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # stem words to make them as their root word
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    fd = nltk.FreqDist(words)
    return [x[0] for x in fd.most_common(10)]

if __name__ == "__main__":

    print("Running program.")

