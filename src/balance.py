from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd

DATASETS_BALANCE_DIR = "./datasets_balance"
DATASETS_BALANCE_DIFFS_DIR = "./datasets_balance_diffs"
DATA_DIR = "./data"


colors_dict = {
    'pants-fire': 'orangered',
    'false': 'coral',
    'barely-true': 'salmon',
    'half-true': 'peachpuff',
    'mostly-true': 'skyblue',
    'true': 'deepskyblue'
}


def autopct_format(values: pd.Series[int]) -> Callable[[float], str]:
    def my_format(pct: float) -> str:
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def add_labels(x: list[str], y: list[float]):
    for i in range(len(x)):
        plt.text(i, y[i] / 2, f'{y[i]}%', ha='center')


def plot_balance(name: str) -> pd.Series[int]:
    global DATA_DIR
    global DATASETS_BALANCE_DIR
    global colors_dict

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
    plt.title(f'{name}.tsv dataset balance')

    colors = [colors_dict[label] for label in label_counts.index]

    plt.pie(label_counts, labels=label_counts.index,
            autopct=autopct_format(label_counts),
            colors=colors)
    plt.savefig(f"{DATASETS_BALANCE_DIR}/{name}.png")
    print(f'{name} dataset plotted correctly!')
    return label_counts


def plot_label_counts_comparison(label_count1: pd.Series[int],
                                 label_count2: pd.Series[int],
                                 name1: str, name2: str) -> None:
    global DATASETS_BALANCE_DIFFS_DIR
    global colors_dict

    diffs = []
    total_count1 = label_count1.sum()
    total_count2 = label_count2.sum()

    for label in label_count1.index:
        diff = round(((label_count1[label] / total_count1) -
                     (label_count2[label] / total_count2)) * 100.0, 2)
        diffs.append(diff)

    plt.cla()
    plt.clf()
    plt.close()
    colors = [colors_dict[label] for label in label_count1.index]
    plt.figure(figsize=(10, 5))
    plt.title(f'{name1} and {name2} dataset balance difference')
    plt.xlabel("label")
    plt.ylabel("percentage difference")
    plt.bar(label_count1.index,
            diffs,
            color=colors)
    add_labels(label_count1.index, diffs)
    plt.savefig(f"{DATASETS_BALANCE_DIFFS_DIR}/{name1}-{name2}.png")
    print(f'{name1}-{name2} dataset difference plotted correctly!')


if __name__ == '__main__':
    test2_label_counts = plot_balance("test2")
    val2_label_counts = plot_balance("val2")
    train2_label_counts = plot_balance("train2")

    plot_label_counts_comparison(
        test2_label_counts,
        val2_label_counts,
        "test2",
        "val2"
    )
    plot_label_counts_comparison(
        test2_label_counts,
        train2_label_counts,
        "test2",
        "train2"
    )
    plot_label_counts_comparison(
        val2_label_counts,
        train2_label_counts,
        "val2",
        "train2"
    )
