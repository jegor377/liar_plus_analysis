from tqdm import tqdm
from transformers import Pipeline, pipeline

from ds_loader import load_dataset


def add_sentiment_column(dataset_name: str,
                         gibbersish_pipeline: Pipeline) -> None:
    df = load_dataset(dataset_name)
    
    gibberish_list = []

    print(f"analyzing gibberish of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement = str(row["statement"])
        gibberish = gibbersish_pipeline(statement)[0]['label']
        gibberish_list.append(gibberish)

    df["gibberish"] = gibberish_list
    df["gibberish"].to_csv(
        f"data/result/gibberish/{dataset_name}.tsv",
        sep="\t",
        index=False)


if __name__ == '__main__':
    pipeline = pipeline(
        "text-classification",
        model="madhurjindal/autonlp-Gibberish-Detector-492513457",
)

    add_sentiment_column("test2", pipeline)
    add_sentiment_column("train2", pipeline)
    add_sentiment_column("val2", pipeline)
