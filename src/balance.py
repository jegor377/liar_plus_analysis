from __future__ import annotations

import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd

DATASETS_BALANCE_DIR = "./datasets_balance"
DATA_DIR = "./data"


def autopct_format(values: pd.Series[int]) -> Callable[[float], str]:
    def my_format(pct: float) -> str:
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def plot_balance(name: str) -> None:
    global DATA_DIR
    global DATASETS_BALANCE_DIR
    dataset = pd.read_csv(f'{DATA_DIR}/{name}.tsv', header=None, sep="\t")
    dataset.columns = [
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
    ]
    label_counts = dataset.label.value_counts()
    plt.cla()
    plt.clf()
    plt.close()

    colors = [
        'orangered',  # pants-fire
        'coral',  # false
        'salmon',  # barely-true
        'peachpuff',  # half-true
        'skyblue',  # mostly-true
        'deepskyblue'  # true
    ]
    labels = [
        'pants-fire',
        'false',
        'barely-true',
        'half-true',
        'mostly-true',
        'true'
    ]

    plt.title(f'{name}.tsv dataset balance')
    plt.pie(label_counts, labels=labels,
            autopct=autopct_format(label_counts),
            colors=colors)
    plt.savefig(f"{DATASETS_BALANCE_DIR}/{name}.png")
    print(f'{name} dataset plotted correctly!')


if __name__ == '__main__':
    if not os.path.isdir(DATASETS_BALANCE_DIR):
        os.mkdir(DATASETS_BALANCE_DIR)
        with open(f'{DATASETS_BALANCE_DIR}/.gitignore', 'w') as f:
            f.write("*\n")

    plot_balance("test2")
    plot_balance("val2")
    plot_balance("train2")
