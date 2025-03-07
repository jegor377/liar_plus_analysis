from tqdm import tqdm
from transformers import Pipeline, pipeline

from ds_loader import load_dataset


def add_sentiment_column(dataset_name: str,
                         emotions_pipeline: Pipeline) -> None:
    df = load_dataset(dataset_name)

    emotions = []

    print(f"analyzing emotions of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement = str(row["statement"])
        emotion = emotions_pipeline(statement)[0]['label']
        emotions.append(emotion)

    df["emotion"] = emotions
    df["emotion"].to_csv(
        f"data/result/emotions/{dataset_name}.tsv",
        sep="\t",
        index=False)


if __name__ == '__main__':
    emotions_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
)

    add_sentiment_column("test2", emotions_pipeline)
    add_sentiment_column("train2", emotions_pipeline)
    add_sentiment_column("val2", emotions_pipeline)
