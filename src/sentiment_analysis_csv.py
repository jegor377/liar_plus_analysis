from __future__ import annotations

from collections import Counter

import pandas as pd
from tqdm import tqdm
from transformers import Pipeline, pipeline

from conf import DATASETS_SENTIMENT_ANALYSIS_CSV_DIR
from ds_loader import load_dataset

SentimentResult = dict[str, dict[str, Counter[str]]]


def analyze_dataset(name: str, pipeline: Pipeline) -> SentimentResult:
    dataset = load_dataset(name)

    sentiments: SentimentResult = {}

    for ttype in ["statement", "justification"]:
        for label in dataset["label"].unique():
            if ttype not in sentiments:
                sentiments[ttype] = {}
            sentiments[ttype][label] = Counter()

    print(f"sentiment analysis on {name} dataset...")

    label2tag = {
        "LABEL_0": 'negative',
        "LABEL_1": 'neutral',
        "LABEL_2": 'positive'
    }

    for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        label = row['label']
        statement = row['statement']
        justification = row['justification']

        (statement_sentiment,
         justification_sentiment) = pipeline([str(statement),
                                              str(justification)],
                                             max_length=512,
                                             truncation=True)

        statement_sentiment = label2tag[statement_sentiment["label"]]
        justification_sentiment = label2tag[justification_sentiment["label"]]

        sentiments["statement"][label][statement_sentiment] += 1
        sentiments["justification"][label][justification_sentiment] += 1

    return sentiments


def save_data(name: str, sentiments: SentimentResult) -> None:
    rows = []

    print(f"saving data from processing {name} dataset...")

    for ttype in sentiments.keys():
        for label in sentiments[ttype].keys():
            sentiment = sentiments[ttype][label]
            row = (
                label,
                ttype,
                sentiment['negative'],
                sentiment['neutral'],
                sentiment['positive']
            )
            rows.append(row)

    df = pd.DataFrame(rows, columns=[
        'label',
        'type',
        'negative',
        'neutral',
        'positive'
    ])
    df.to_csv(
        f"{DATASETS_SENTIMENT_ANALYSIS_CSV_DIR}/{name}.tsv",
        sep='\t',
        index=False
    )


def process_dataset(name: str, pipeline: Pipeline) -> None:
    sentiments = analyze_dataset(name, pipeline)
    save_data(name, sentiments)


if __name__ == "__main__":
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment")

    process_dataset("test2", sentiment_pipeline)
    process_dataset("train2", sentiment_pipeline)
    process_dataset("val2", sentiment_pipeline)
