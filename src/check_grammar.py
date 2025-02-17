from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable

import language_tool_python
import matplotlib.pyplot as plt
import tqdm

from conf import (DATASETS_GRAMMAR_CHECK_CSV_DIR,
                  DATASETS_GRAMMAR_CHECK_PLOT_DIR, label_colors)
from ds_loader import load_dataset


def autopct_format(values: list[int]) -> Callable[[float], str]:
    def my_format(pct: float) -> str:
        total = sum(values)
        val = int(pct*total/100.0)
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def create_csv_report(dataset_name: str,
                      error_ids: list[str],
                      statement_errors: dict[str, Counter[str]],
                      justification_errors: dict[str, Counter[str]]) -> None:
    labels: list[str] = [
        'pants-fire',
        'false',
        'barely-true',
        'half-true',
        'mostly-true',
        'true'
    ]

    print('creating csv for {dataset_name} dataset...')
    csv_filepath = f"{DATASETS_GRAMMAR_CHECK_CSV_DIR}/{dataset_name}.csv"
    with open(csv_filepath, 'w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["error_id"] +
                        [f'statement_{label}' for label in labels] +
                        [f'justification_{label}' for label in labels])

        for error_id in tqdm.tqdm(error_ids):
            row = [error_id]

            for label in labels:
                if label not in statement_errors.keys():
                    row.append(0)
                else:
                    row.append(statement_errors[label][error_id])

            for label in labels:
                if label not in justification_errors.keys():
                    row.append(0)
                else:
                    row.append(justification_errors[label][error_id])

            writer.writerow(row)


def create_plots(dataset_name: str,
                 statement_errors: dict[str, Counter[str]],
                 justification_errors: dict[str, Counter[str]]) -> None:
    statement_sums: Counter[str] = Counter()
    justification_sums: Counter[str] = Counter()

    print(f"plotting dataset {dataset_name}...")

    for label in statement_errors.keys():
        statement_sums[label] = sum(statement_errors[label].values())

    for label in justification_errors.keys():
        justification_sums[label] = sum(justification_errors[label].values())

    plt.cla()
    plt.clf()
    plt.close()
    fig, axis = plt.subplots(1, 2)
    axis[0].pie(statement_sums.values(),
                labels=list(statement_sums.keys()),
                autopct=autopct_format(list(statement_sums.values())),
                colors=label_colors(list(statement_sums.keys())),
                textprops={'fontsize': 8})
    axis[0].title.set_text("statements")
    axis[1].pie(justification_sums.values(),
                labels=list(justification_sums.keys()),
                autopct=autopct_format(list(justification_sums.values())),
                colors=label_colors(list(justification_sums.keys())),
                textprops={'fontsize': 8})
    axis[1].title.set_text("justifications")
    fig.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.85)
    fig.suptitle(f"grammatical errors in {dataset_name}", y=0.9)
    plt.savefig(
        f"{DATASETS_GRAMMAR_CHECK_PLOT_DIR}/{dataset_name}.png", dpi=400)
    print(f'{dataset_name} dataset plotted correctly!')


def process_dataset(dataset_name: str,
                    tool: language_tool_python.LanguageTool) -> None:
    statement_errors: dict[str, Counter[str]] = {}
    justification_errors: dict[str, Counter[str]] = {}

    error_ids: list[str] = []

    dataset = load_dataset(dataset_name)

    print(f"checking grammar for {dataset_name} dataset...")
    for _, row in tqdm.tqdm(dataset.iterrows()):
        label = row.label
        statement = row.statement
        justification = row.justification

        statement_matches = tool.check(statement)
        justification_matches = tool.check(justification)

        for match in statement_matches:
            if label not in statement_errors:
                statement_errors[label] = Counter()
            statement_errors[label][match.ruleId] += 1

            if match.ruleId not in error_ids:
                error_ids.append(match.ruleId)

        for match in justification_matches:
            if label not in justification_errors:
                justification_errors[label] = Counter()
            justification_errors[label][match.ruleId] += 1

            if match.ruleId not in error_ids:
                error_ids.append(match.ruleId)

    error_ids.sort()

    create_csv_report(dataset_name, error_ids,
                      statement_errors, justification_errors)
    create_plots(dataset_name, statement_errors, justification_errors)


if __name__ == '__main__':
    tool = language_tool_python.LanguageTool('en-US')

    process_dataset("test2", tool)
    process_dataset("train2", tool)
    process_dataset("val2", tool)
