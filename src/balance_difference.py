from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from balance_misc.balance_conf import DATASETS_BALANCE_DIFFS_DIR, label_colors
from balance_misc.ds_loader import load_dataset


def add_labels(x: list[str], y: list[float]):
    for i in range(len(x)):
        plt.text(i, y[i] / 2, f'{y[i]}%', ha='center')


def plot_label_counts_comparison(label_count1: pd.Series[int],
                                 label_count2: pd.Series[int],
                                 name1: str, name2: str) -> None:
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
    plt.figure(figsize=(10, 5))
    plt.title(f'{name1} and {name2} dataset balance difference')
    plt.xlabel("label")
    plt.ylabel("percentage difference")
    plt.bar(label_count1.index,
            diffs,
            color=label_colors(label_count1.index))
    add_labels(label_count1.index, diffs)
    plt.savefig(f"{DATASETS_BALANCE_DIFFS_DIR}/{name1}-{name2}.png")
    print(f'{name1}-{name2} dataset difference plotted correctly!')


if __name__ == '__main__':
    test2_label_counts = load_dataset('test2')['label'].value_counts()
    val2_label_counts = load_dataset('val2')['label'].value_counts()
    train2_label_counts = load_dataset('train2')['label'].value_counts()

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
