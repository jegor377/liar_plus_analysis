from __future__ import annotations

import matplotlib.pyplot as plt

from balance_misc.balance_conf import DATASETS_BALANCE_MEDIAN_DIR
from balance_misc.ds_loader import load_dataset


def plot_boxplot(names: list[str]) -> None:
    labels_counts = []

    for name in names:
        label_counts = load_dataset(name)['label'].value_counts()
        label_counts = label_counts / label_counts.sum() * 100.0
        labels_counts.append(label_counts)

    plt.cla()
    plt.clf()
    plt.close()
    plt.title(f'{", ".join(names)} datasets balance')
    plt.ylabel("percentage of whole dataset")
    plt.boxplot(labels_counts)
    plt.xticks(list(range(len(names))), names)
    for x in range(len(names)):
        for y in labels_counts[x]:
            plt.plot(x + 1, y, 'r.', alpha=0.5)
    plt.savefig(f"{DATASETS_BALANCE_MEDIAN_DIR}/boxplot.png")
    print('boxplot plotted correctly!')


if __name__ == '__main__':
    plot_boxplot(["test2", "val2", "train2"])
