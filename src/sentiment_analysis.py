from __future__ import annotations

from collections import Counter

from tqdm import tqdm
from transformers import Pipeline, pipeline

from ds_loader import load_dataset

SentimentResult = tuple[Counter[str], Counter[str]]


def process_dataset(name: str, pipeline: Pipeline) -> SentimentResult:
    dataset = load_dataset(name)

    statement_sentiments: Counter[str] = Counter()
    justification_sentiments: Counter[str] = Counter()

    print(f"sentiment analysis on {name} dataset...")

    for _, row in tqdm(dataset.iterrows()):
        statement = row['statement']
        justification = row['justification']

        (statement_sentiment,
         justification_sentiment) = pipeline([str(statement),
                                              str(justification)],
                                             max_length=512,
                                             truncation=True)

        statement_sentiments[statement_sentiment["label"]] += 1
        justification_sentiments[justification_sentiment["label"]] += 1

    return (statement_sentiments, justification_sentiments)


if __name__ == "__main__":
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment")

    print(process_dataset("test2", sentiment_pipeline))
