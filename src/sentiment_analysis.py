import pandas as pd
from tqdm import tqdm
from transformers import Pipeline, pipeline


def add_sentiment_column(dataset_name: str,
                         sentiment_pipeline: Pipeline) -> None:
    df = pd.read_csv(f"data/result/headers/{dataset_name}.tsv", sep='\t')

    label2tag = {
        "LABEL_0": 'negative',
        "LABEL_1": 'neutral',
        "LABEL_2": 'positive'
    }
    sentiments = []

    print(f"analysing sentiment of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement = str(row["statement"])
        sentiment = sentiment_pipeline(statement)[0]['label']
        sentiments.append(label2tag[sentiment])

    df["sentiment"] = sentiments
    df.to_csv(
        f"data/result/sentiment/{dataset_name}.tsv",
        sep="\t",
        index=False)


if __name__ == '__main__':
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        truncation=True,
        max_length=512)
    add_sentiment_column("test2", sentiment_pipeline)
    add_sentiment_column("train2", sentiment_pipeline)
    add_sentiment_column("val2", sentiment_pipeline)
