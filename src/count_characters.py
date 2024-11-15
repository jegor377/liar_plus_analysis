from __future__ import annotations

import string
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd

from conf import DATASETS_COUNTS_DIR, label_colors
from ds_loader import load_dataset


def autopct_format(values: pd.Series[int]) -> Callable[[float], str]:
    def my_format(pct: float) -> str:
        total = sum(values)
        val = pct*total/100.0
        return '{:.1f}%\n({v:.4f})'.format(pct, v=val)
    return my_format


def count_punctuations(name: str) -> None:
    dataset = load_dataset(name)

    statements_by_label = {}
    justification_by_label = {}

    for _, row in dataset.iterrows():
        label = row["label"]
        statement = row["statement"]
        justification = row["justification"]

        if not type(label) is str:
            continue

        if label not in statements_by_label:
            statements_by_label[label] = {
                "punctuations": 0,
                "total": 0
            }
        if label not in justification_by_label:
            justification_by_label[label] = {
                "punctuations": 0,
                "total": 0
            }

        if type(statement) is str:
            statements_by_label[label]["punctuations"] += sum(
                map(statement.count, string.punctuation))
            statements_by_label[label]["total"] += len(statement)

        if type(justification) is str:
            justification_by_label[label]["punctuations"] += sum(
                map(justification.count, string.punctuation))
            justification_by_label[label]["total"] += len(justification)

    statements = {
        k: x["punctuations"] / x["total"] * 100.0
        for k, x in statements_by_label.items()
    }
    justifications = {
        k: x["punctuations"] / x["total"] * 100.0
        for k, x in justification_by_label.items()
    }

    statement_labels = list(statements.keys())
    justification_labels = list(justifications.keys())
    statement_values = list(statements.values())
    justification_values = list(justifications.values())

    plt.cla()
    plt.clf()
    plt.close()
    fig, axis = plt.subplots(1, 2)
    axis[0].pie(statement_values, labels=statement_labels,
                autopct=autopct_format(statement_values),
                colors=label_colors(statement_labels),
                textprops={'fontsize': 8})
    axis[0].title.set_text("statements")
    axis[1].pie(justification_values, labels=justification_labels,
                autopct=autopct_format(justification_values),
                colors=label_colors(justification_labels),
                textprops={'fontsize': 8})
    axis[1].title.set_text("justifications")
    fig.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.85)
    fig.suptitle("punctuations per 100 characters", y=0.9)
    plt.savefig(f"{DATASETS_COUNTS_DIR}/{name}.png", dpi=400)
    print(f'{name} dataset plotted correctly!')


if __name__ == '__main__':
    count_punctuations("test2")
    count_punctuations("val2")
    count_punctuations("train2")
