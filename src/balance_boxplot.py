from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from balance_misc.conf import (DATASETS_BALANCE_MEDIAN_DIR, colors_dict,
                               label_colors)
from ds_loader import load_dataset


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
    plt.ylabel("% of whole dataset")
    plt.boxplot(labels_counts)
    plt.xticks(list(range(len(names))), names)
    for x in range(len(names)):
        for i, y in enumerate(labels_counts[x]):
            plt.scatter(x + 1, y, c=label_colors(labels_counts[x].index)[i])
    patches = []
    labels = list(colors_dict.keys())
    colors = label_colors(labels)
    for i, label in enumerate(labels):
        patches.append(mpatches.Patch(color=colors[i], label=label))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', handles=patches)
    plt.tight_layout()
    plt.savefig(f"{DATASETS_BALANCE_MEDIAN_DIR}/boxplot.png")
    print('boxplot plotted correctly!')


if __name__ == '__main__':
    plot_boxplot(["test2", "val2", "train2"])
