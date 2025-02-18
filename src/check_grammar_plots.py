from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable

import matplotlib.pyplot as plt
import tqdm

from conf import (DATASETS_GRAMMAR_CHECK_CSV_DIR,
                  DATASETS_GRAMMAR_CHECK_PLOT_DIR, label_colors)


def autopct_format(values: list[int]) -> Callable[[float], str]:
    def my_format(pct: float) -> str:
        total = sum(values)
        val = int(pct*total/100.0)
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def create_plots(dataset_name: str,
                 statement_errors: Counter[str],
                 justification_errors: Counter[str]) -> None:

    print(f"plotting dataset {dataset_name}...")

    plt.cla()
    plt.clf()
    plt.close()
    fig, axis = plt.subplots(1, 2)
    axis[0].pie(statement_errors.values(),
                labels=list(statement_errors.keys()),
                autopct=autopct_format(list(statement_errors.values())),
                colors=label_colors(list(statement_errors.keys())),
                textprops={'fontsize': 8})
    axis[0].title.set_text("statements")
    axis[1].pie(justification_errors.values(),
                labels=list(justification_errors.keys()),
                autopct=autopct_format(list(justification_errors.values())),
                colors=label_colors(list(justification_errors.keys())),
                textprops={'fontsize': 8})
    axis[1].title.set_text("justifications")
    fig.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.85)
    fig.suptitle(f"grammatical errors in {dataset_name}", y=0.9)
    plt.savefig(
        f"{DATASETS_GRAMMAR_CHECK_PLOT_DIR}/{dataset_name}.png", dpi=400)
    print(f'{dataset_name} dataset plotted correctly!')


def process_result(dataset_name: str) -> None:
    statement_errors: Counter[str] = Counter()
    justification_errors: Counter[str] = Counter()

    print(f"loading csv data of {dataset_name} dataset...")

    filepath = DATASETS_GRAMMAR_CHECK_CSV_DIR + f"/{dataset_name}.csv"
    with open(filepath, 'r', encoding='utf-8', newline='\n') as f:
        reader = csv.reader(f, delimiter='\t')

        # skip header
        header = next(reader)

        for row in tqdm.tqdm(reader):
            for i in range(1, len(header)):
                if 'statement' in header[i]:
                    statement_errors[header[i].replace(
                        'statement_', '')] += int(row[i])
                else:
                    justification_errors[header[i].replace(
                        'justification_', '')] += int(row[i])

        create_plots(dataset_name, statement_errors, justification_errors)


if __name__ == '__main__':
    process_result("test2")
    process_result("train2")
    process_result("val2")
