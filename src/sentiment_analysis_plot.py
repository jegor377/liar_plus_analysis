from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from conf import (DATASETS_SENTIMENT_ANALYSIS_CSV_DIR,
                  DATASETS_SENTIMENT_ANALYSIS_PLOT_DIR)


def plot(name: str) -> None:
    df = pd.read_csv(
        f"{DATASETS_SENTIMENT_ANALYSIS_CSV_DIR}/{name}.tsv",
        sep='\t'
    )

    types = df["type"].unique()
    labels = [
        'pants-fire',
        'false',
        'barely-true',
        'half-true',
        'mostly-true',
        'true'
    ]

    negative: dict[str, list[int]] = {
        "statement": [],
        "justification": []
    }
    neutral: dict[str, list[int]] = {
        "statement": [],
        "justification": []
    }
    positive: dict[str, list[int]] = {
        "statement": [],
        "justification": []
    }

    for ttype in types:
        for label in labels:
            sentiment = df[(df["type"] == ttype) & (
                df["label"] == label)].iloc[0]
            negative[ttype].append(sentiment["negative"])
            neutral[ttype].append(sentiment["neutral"])
            positive[ttype].append(sentiment["positive"])

    x = np.arange(len(labels))
    bar_width = 0.25

    plt.cla()
    plt.clf()
    plt.close()
    fig, axis = plt.subplots(1, 2, figsize=(10, 5))

    axis[0].bar(
        x - bar_width,
        negative["statement"],
        bar_width,
        label="negative",
        color="skyblue"
    )
    axis[0].bar(
        x,
        neutral["statement"],
        bar_width,
        label="neutral",
        color="lightgreen"
    )
    axis[0].bar(
        x + bar_width,
        positive["statement"],
        bar_width,
        label="positive",
        color="darkorange"
    )

    axis[0].set_ylabel("number of rows")
    axis[0].set_xticks(
        x, labels,
        rotation=45
    )

    axis[1].bar(
        x - bar_width,
        negative["justification"],
        bar_width,
        label="negative",
        color="skyblue"
    )
    axis[1].bar(
        x,
        neutral["justification"],
        bar_width,
        label="neutral",
        color="lightgreen"
    )
    axis[1].bar(
        x + bar_width,
        positive["justification"],
        bar_width,
        label="positive",
        color="darkorange"
    )

    axis[1].set_ylabel("number of rows")
    axis[1].set_xticks(
        x, labels,
        rotation=45
    )

    axis[0].title.set_text("statements")
    axis[1].title.set_text("justifications")
    fig.tight_layout(pad=3)
    fig.suptitle(f"sentiments in {name}")
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines: plt.Artist
    lines, labels = [sum(ll, []) for ll in zip(*lines_labels)]
    fig.legend(lines[:3], labels[:3], loc='upper right')
    plt.subplots_adjust(left=0.1, right=0.85, bottom=0.2)
    plt.savefig(
        f"{DATASETS_SENTIMENT_ANALYSIS_PLOT_DIR}/{name}.png", dpi=400)
    print(f'{name} dataset plotted correctly!')


if __name__ == '__main__':
    plot("test2")
    plot("train2")
    plot("val2")
