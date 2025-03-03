from tqdm import tqdm
from transformers import Pipeline, pipeline

from ds_loader import load_dataset


def add_sentiment_column(dataset_name: str,
                         sentiment_pipeline: Pipeline) -> None:
    df = load_dataset(dataset_name)

    sentiments = []

    print(f"analysing sentiment of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement = str(row["statement"])
        sentiment = sentiment_pipeline(statement)[0]['label']
        sentiments.append(sentiment)

    df["sentiment"] = sentiments
    df["sentiment"].to_csv(
        f"data/result/sentiment/{dataset_name}.tsv",
        sep="\t",
        index=False)


if __name__ == '__main__':
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        truncation=True,
        max_length=512)
    sentiment_pipeline.model.config.id2label = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }

    add_sentiment_column("test2", sentiment_pipeline)
    add_sentiment_column("train2", sentiment_pipeline)
    add_sentiment_column("val2", sentiment_pipeline)
