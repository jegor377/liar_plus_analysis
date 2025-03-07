from tqdm import tqdm
from transformers import Pipeline, pipeline

from ds_loader import load_dataset


def add_sentiment_column(dataset_name: str,
                         emotions_pipeline: Pipeline) -> None:
    df = load_dataset(dataset_name)
    
    biases = []

    print(f"analyzing political bias of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        statement = str(row["statement"])
        bias = emotions_pipeline(statement)[0]['label']
        biases.append(bias)

    df["political_bias"] = biases
    df["political_bias"].to_csv(
        f"data/result/political_bias/{dataset_name}.tsv",
        sep="\t",
        index=False)


if __name__ == '__main__':
    pipeline = pipeline(
        "text-classification",
        model="bucketresearch/politicalBiasBERT",
)

    add_sentiment_column("test2", pipeline)
    add_sentiment_column("train2", pipeline)
    add_sentiment_column("val2", pipeline)
