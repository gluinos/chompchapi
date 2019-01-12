import pandas as pd

# pd.set_option('display.expand_frame_repr', False)

def get_combined_df(fname_reviews, fname_businesses):
    """
    Takes filenames for review and business kaggle dataset JSONs and returns
    a pandas dataframe of their combined information
    Reviews: https://www.kaggle.com/yelp-dataset/yelp-dataset#yelp_academic_dataset_review.json
    Businesses: https://www.kaggle.com/yelp-dataset/yelp-dataset#yelp_academic_dataset_business.json
    """
    df_reviews = pd.read_json(fname_reviews,lines=True).drop(columns=["review_id","user_id"])
    df_businesses = pd.read_json(fname_businesses,lines=True).drop(columns=["address","city","latitude","longitude","neighborhood","postal_code","state","hours","is_open"])
    df = df_reviews.merge(df_businesses, on=["business_id"], how="left")
    return df

def skim_df(df):
    """
    Some basic cuts here
    """
    return df[(df.review_count>150) & (df.stars_y > 2.5) & (df.stars_x > 2.5)]


if __name__ == "__main__":

    df = get_combined_df(
            "yelp_academic_dataset_reviews.json",
            "yelp_academic_dataset_business.json",
            )
    df = skim_df(df)
    print(df.shape)
    df.to_pickle("combined_data.pkl")
    # df = pd.read_pickle("combined_data.pkl")
