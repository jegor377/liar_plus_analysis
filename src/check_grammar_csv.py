from __future__ import annotations

import csv
from collections import Counter

import language_tool_python
import tqdm

from conf import DATASETS_GRAMMAR_CHECK_CSV_DIR
from ds_loader import load_dataset


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

    print(f'creating csv for {dataset_name} dataset...')
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


def process_dataset(dataset_name: str,
                    tool: language_tool_python.LanguageTool) -> None:
    statement_errors: dict[str, Counter[str]] = {}
    justification_errors: dict[str, Counter[str]] = {}

    error_ids: list[str] = []

    dataset = load_dataset(dataset_name)

    print(f"checking grammar for {dataset_name} dataset...")
    for _, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
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


if __name__ == '__main__':
    tool = language_tool_python.LanguageTool('en-US')

    process_dataset("test2", tool)
    process_dataset("train2", tool)
    process_dataset("val2", tool)
