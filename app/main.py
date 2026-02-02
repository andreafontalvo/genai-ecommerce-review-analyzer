from data_loader import load_reviews
from sentiment import run_sentiment
from bedrock import extract_pros_cons, summarize

CLOTHING_ID = 1110
SAMPLE_NUM = 300
CHUNK_SIZE = 10

def main():
    reviews = load_reviews(CLOTHING_ID, SAMPLE_NUM)
    sentiment_df, pos_rate, neg_rate = run_sentiment(reviews)
    chunks = [sentiment_df[i:i+CHUNK_SIZE] for i in range(0, len(sentiment_df), CHUNK_SIZE)]
    
    results = []

    for chunk in chunks:
     results.append(extract_pros_cons(chunk))

    pros, cons = [], []
    for r in results:
        pros.extend(r.get("pros", []))
        cons.extend(r.get("cons", []))

    report = summarize(pros, cons)

    print("Review Analysis")
    print(f"Positive: {pos_rate}%")
    print(f"Negative: {neg_rate}%")
    print(report)


if __name__ == "__main__":
    main()