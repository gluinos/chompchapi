import pandas as pd
import numpy as np

from keras.models import Model
from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from tqdm import tqdm

from sklearn.cluster import KMeans

from collections import Counter


if __name__ == "__main__":

    df = pd.read_json("yelp_academic_dataset_business.json",lines=True).filter(["categories"])
    # df = pd.read_json("small.json",lines=True).filter(["categories"])
    df = df[df.categories.str.contains("Restaurants", na=False)]
    df.reset_index(inplace=True)
    # print df.categories.str.split(",")
    print df.categories.str.count(",")
    # df = pd.read_pickle("combined_data.pkl")
    # df = df[:5000]

    # tok = Tokenizer(num_words=max_words)
    # tok.fit_on_texts(X_train)
    # sequences = tok.texts_to_sequences(X_train)
    # sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

    all_categs_2d = []
    all_categs_2d_flat = []
    max_num_categs = 13
    for categs in tqdm(df.categories.str.split(",")):
        if not categs: categs = []
        nice_categs = []
        for c in categs:
            c = c.strip()
            if str(c) in ["Restaurants","Food"]:
                continue
            nice_categs.append(c)
        nice_categs = sorted(nice_categs)[:max_num_categs]
        # nice_categs = sorted(map(lambda x:x.strip(),categs))[:max_num_categs]
        max_num_categs = max(len(nice_categs), max_num_categs)
        all_categs_2d.append(nice_categs)
        all_categs_2d_flat.extend(nice_categs)
    all_categs_2d = np.array(all_categs_2d)
    all_categs_2d_flat = sorted(list(set(all_categs_2d_flat)))
    ncategs = len(all_categs_2d_flat)
    categ_idxs = {categ:i+1 for i,categ in enumerate(all_categs_2d_flat)}

    vecs = []
    for categs in all_categs_2d:
        vec = [0 for _ in range(max_num_categs)]
        for icateg,categ in enumerate(categs):
            vec[icateg] = categ_idxs[categ]
        vecs.append(vec)
    vecs = np.array(vecs)
    print vecs

    print ncategs
        

    # tok = Tokenizer(num_words=ncategs,split=",")
    # print ",".join(all_categs_2d_flat)
    # tok.fit_on_texts(",".join(all_categs_2d_flat))
    # sequences = tok.texts_to_sequences(all_categs_2d)
    # sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_num_categs)
    

    model = Sequential()
    model.add(Embedding(ncategs+1, 8))
    # input_array = np.random.randint(5, size=(1, 5))
    input_array = vecs

    model.compile('rmsprop', 'mse')
    output_array = model.predict(input_array)
    print output_array
    print output_array.shape

    # Now average over the words
    feat_vecs = output_array.mean(1)
    print feat_vecs.shape

    kmeans = KMeans(n_clusters=15, random_state=0).fit(feat_vecs)
    print kmeans.labels_

    def get_common(arr):
        flat = []
        for a in arr: flat.extend(a)
        return sorted(dict(Counter(flat)).items(),key=lambda x:-x[1])[:20]

    for i in range(15):
        print "---- {} ----".format(i)
        arr = all_categs_2d[kmeans.labels_  == i]
        N = len(arr)
        freqs = get_common(arr)
        for c,num in freqs:
            print "   {:<25s} {:.2f}".format(c,1.0*num/N)
        print ",".join('"{}"'.format(c) for c,num in freqs)
        # print "\n".join(get_common(arr))

    # print "0", all_categs_2d[kmeans.labels_  == 0]
    # print "1", all_categs_2d[kmeans.labels_  == 1]
    # print "2", all_categs_2d[kmeans.labels_  == 2]
    # print "3", all_categs_2d[kmeans.labels_  == 3]
    # print "4", all_categs_2d[kmeans.labels_  == 4]
