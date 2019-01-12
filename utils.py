import json
import nltk
import pandas as pd


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


if __name__ == "__main__":
    json_path = "/"
    df = load_json(json_path)
    for item in df:
        tokenize(item)
