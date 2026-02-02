import pandas as pd
import numpy as np
from transformers import pipeline

pipe = pipeline(
"sentiment-analysis",
model="distilbert-base-uncased-finetuned-sst-2-english"
)


def run_sentiment(reviews):

    results = pipe(reviews, batch_size=32)
    df = pd.DataFrame(results)

    counts = df["label"].value_counts()

    positive = round(counts.get("POSITIVE", 0) / len(df) * 100)
    negative = 100 - positive

    out = pd.DataFrame({
        "Review Text": reviews,
        "Sentiment": df["label"]
        })

    return out, positive, negative