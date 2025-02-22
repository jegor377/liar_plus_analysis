import numpy as np
import matplotlib.pyplot as plt
from ds_loader import load_dataset
from conf import (DATASETS_SENTENCE_LENGTH_PLOT_DIR, label_colors)


def process_dataset(dataset_name: str) -> None:
    print(f"processing dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)

    lengths_for_labels: dict[str, list[int]] = {}
    lengths_for_labels["pants-fire"] = []
    lengths_for_labels["false"] = []
    lengths_for_labels["barely-true"] = []
    lengths_for_labels["half-true"] = []
    lengths_for_labels["mostly-true"] = []
    lengths_for_labels["true"] = []

    for _, row in dataset.iterrows():
        label = row.label
        statement = row.statement
        lengths_for_labels[label].append(len(statement))
        
    avg_results = {key: np.mean(lengths_for_labels[key]) for key in lengths_for_labels}
    median_results = {key: np.median(lengths_for_labels[key]) for key in lengths_for_labels}

    fig, axis = plt.subplots(1, 2)
    colors_avg = label_colors(list(avg_results.keys()))
    colors_median = label_colors(list(median_results.keys()))

    axis[0].bar(avg_results.keys(), avg_results.values(), color=colors_avg)
    axis[0].set_title("average sentence length")
    axis[0].set_xticks(range(len(avg_results.keys())))
    axis[0].set_xticklabels(avg_results.keys(), rotation=90)
    axis[0].set_ylim([0, 130])

    for i, v in enumerate(avg_results.values()):
        axis[0].text(i, v + 2, f"{v:.1f}", ha='center', fontsize=8, fontweight='bold')

    axis[1].bar(median_results.keys(), median_results.values(), color=colors_median)
    axis[1].set_title("median sentence length")
    axis[1].set_xticks(range(len(median_results.keys())))
    axis[1].set_xticklabels(median_results.keys(), rotation=90)
    axis[1].set_ylim([0, 130])

    for i, v in enumerate(median_results.values()):
        axis[1].text(i, v + 2, f"{v:.1f}", ha='center', fontsize=8, fontweight='bold')

    fig.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.85)
    fig.suptitle(f"sentence length in {dataset_name}", y=0.95)
    plt.savefig(
        f"{DATASETS_SENTENCE_LENGTH_PLOT_DIR}/{dataset_name}.png", dpi=400)

if __name__ == '__main__':
    process_dataset("test2")
    process_dataset("train2")
    process_dataset("val2")