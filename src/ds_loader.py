from __future__ import annotations

import pandas as pd

from conf import DATA_DIR


def load_dataset(dataset_name: str) -> pd.DataFrame:
    dataset = pd.read_csv(f'{DATA_DIR}/{dataset_name}.tsv', sep="\t", names=[
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
    return dataset
