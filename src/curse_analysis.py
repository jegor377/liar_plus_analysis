from tqdm import tqdm
from transformers import Pipeline, pipeline

from ds_loader import load_dataset


def add_sentiment_column(dataset_name: str,
                         curse_pipeline: Pipeline) -> None:
    df = load_dataset(dataset_name)

    curses = []

    print(f"analyzing curses of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement = str(row["statement"])
        curse = curse_pipeline(statement)[0]['label']
        curses.append(curse)

    df["curse"] = curses
    df["curse"].to_csv(
        f"data/result/curse/{dataset_name}.tsv",
        sep="\t",
        index=False)


if __name__ == '__main__':
    curse_pipeline = pipeline(
        "text-classification",
        model="djsull/curse_classification",
)

    add_sentiment_column("test2", curse_pipeline)
    add_sentiment_column("train2", curse_pipeline)
    add_sentiment_column("val2", curse_pipeline)
