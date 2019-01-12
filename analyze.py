import pandas as pd
import numpy as np

from reduction import num_metacategories

if __name__ == "__main__":

    df = pd.read_pickle("combined_data.pkl")

    # print df.head()

    print (df.text)

    # Unpack column by column into an num_review-by-num_metacategories matrix again
    target_vecs = np.vstack([
        df["cat_{}".format(i)] for i in range(num_metacategories)
        ]).T
    print (target_vecs)

