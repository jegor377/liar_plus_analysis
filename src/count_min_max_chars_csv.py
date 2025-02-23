from __future__ import annotations

import pandas as pd

from conf import DATASETS_COUNT_MIN_MAX_CHARS_CSV_DIR
from ds_loader import load_dataset


def process_dataset(name: str) -> None:
    dataset = load_dataset(name)[["id", "label", "statement", "justification"]]

    result_rows = []

    labels = dataset["label"].unique()

    dataset["statement_len"] = dataset["statement"].str.len()
    dataset["justification_len"] = dataset["justification"].str.len()

    for ttype in ["statement", "justification"]:
        for label in labels:
            cond1 = (dataset["label"] == label)
            cond2 = (dataset[cond1][f"{ttype}_len"]
                     == dataset[cond1][f"{ttype}_len"].min())
            cond3 = (dataset[cond1][f"{ttype}_len"]
                     == dataset[cond1][f"{ttype}_len"].max())
            min_row = dataset[cond1 & cond2].iloc[0]
            max_row = dataset[cond1 & cond3].iloc[0]

            result_rows.append((
                label,
                ttype,
                min_row[f"{ttype}_len"],
                max_row[f"{ttype}_len"],
                min_row.id,
                max_row.id
            ))

    result = pd.DataFrame(result_rows, columns=[
        "label",
        "type",
        "min",
        "max",
        "min_id",
        "max_id"
    ])

    result.to_csv(
        f"{DATASETS_COUNT_MIN_MAX_CHARS_CSV_DIR}/{name}.tsv",
        sep="\t", mode="w")


if __name__ == "__main__":
    process_dataset("test2")
    process_dataset("train2")
    process_dataset("val2")
