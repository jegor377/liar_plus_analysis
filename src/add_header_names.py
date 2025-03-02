import pandas as pd


def add_headers(dataset_name: str) -> None:
    df = pd.read_csv(f"data/{dataset_name}.tsv", sep="\t", names=[
        "id",
        "json_id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state",
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
        "justification"
    ])

    # drop useless id column
    df = df.drop("id", axis=1)

    df.to_csv(
        f"data/result/headers/{dataset_name}.tsv",
        sep='\t',
        index=False)


if __name__ == '__main__':
    add_headers("test2")
    add_headers("train2")
    add_headers("val2")
