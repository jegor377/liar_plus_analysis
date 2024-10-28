from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd

from balance_misc.balance_conf import DATASETS_BALANCE_DIR, label_colors
from balance_misc.ds_loader import load_dataset


def autopct_format(values: pd.Series[int]) -> Callable[[float], str]:
    def my_format(pct: float) -> str:
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def plot_balance(name: str) -> None:
    label_counts = load_dataset(name)['label'].value_counts()
    plt.cla()
    plt.clf()
    plt.close()
    plt.title(f'{name}.tsv dataset balance')
    plt.pie(label_counts, labels=label_counts.index,
            autopct=autopct_format(label_counts),
            colors=label_colors(list(label_counts.index)))
    plt.savefig(f"{DATASETS_BALANCE_DIR}/{name}.png")
    print(f'{name} dataset plotted correctly!')


if __name__ == '__main__':
    plot_balance("test2")
    plot_balance("val2")
    plot_balance("train2")
