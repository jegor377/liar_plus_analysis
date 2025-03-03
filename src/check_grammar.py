from collections import Counter

import pandas as pd
from language_tool_python import LanguageTool
from tqdm import tqdm


def add_grammar_check_column(dataset_name: str,
                             tool: LanguageTool) -> None:
    df = pd.read_csv(f"data/result/question/{dataset_name}.tsv", sep="\t")

    grammar_errors_count = []
    errors: Counter[str] = Counter()

    print(f"checking grammar for {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        gramatical_errors = tool.check(str(row["statement"]))

        grammar_errors_count.append(len(gramatical_errors))
        for match in gramatical_errors:
            errors[match.ruleId] += 1

    df["grammar_errors"] = grammar_errors_count

    error_df = pd.DataFrame.from_dict(errors, orient='index').reset_index()
    error_df.columns = ["rule_id", "count"]
    error_df.to_csv(f"data/result/grammar/errors/{dataset_name}.tsv",
                    sep="\t", index=False)

    df.to_csv(f"data/result/grammar/{dataset_name}.tsv",
              sep="\t", index=False)


if __name__ == '__main__':
    tool = LanguageTool('en-US')

    add_grammar_check_column("test2", tool)
    add_grammar_check_column("train2", tool)
    add_grammar_check_column("val2", tool)
