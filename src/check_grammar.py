from __future__ import annotations

from collections import Counter

import language_tool_python

from ds_loader import load_dataset

if __name__ == '__main__':
    statement_errors: dict[str, Counter[str]] = {}
    context_errors: dict[str, Counter[str]] = {}
    dataset = load_dataset("test2")

    with language_tool_python.LanguageTool('en-US') as tool:
        for _, row in dataset.iterrows():
            label = row.label
            statement = row.statement
            context = row.context

            statement_matches = tool.check(statement)
            context_matches = tool.check(context)

            for match in statement_matches:
                if label not in statement_errors:
                    statement_errors[label] = Counter()
                statement_errors[label][match.ruleId] += 1

            for match in context_matches:
                if label not in context_errors:
                    context_errors[label] = Counter()
                context_errors[label][match.ruleId] += 1
