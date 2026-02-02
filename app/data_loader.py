import pandas as pd

DATA_PATH = "data/Womens-Clothing-E-Commerce-Reviews.csv"

def load_reviews(clothing_id, sample_size):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Review Text"])
    product = df[df["Clothing ID"] == clothing_id]
    sample = product.sample(sample_size, random_state=23)
    return sample["Review Text"].tolist()