from tqdm import tqdm
from transformers import Pipeline, pipeline

from ds_loader import load_dataset


def add_sentiment_column(dataset_name: str,
                         emotions_pipeline: Pipeline) -> None:
    df = load_dataset(dataset_name)
    
    offensiveness = []

    print(f"analyzing offensiveness of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement = str(row["statement"])
        offensive = emotions_pipeline(statement)[0]['label']
        offensiveness.append(offensive)

    df["offensiveness"] = offensiveness
    df["offensiveness"].to_csv(
        f"data/result/offensive/{dataset_name}.tsv",
        sep="\t",
        index=False)


if __name__ == '__main__':
    pipeline = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-offensive",
)

    add_sentiment_column("test2", pipeline)
    add_sentiment_column("train2", pipeline)
    add_sentiment_column("val2", pipeline)
