from collections import Counter

import pandas as pd
from language_tool_python import LanguageTool
from tqdm import tqdm


def add_grammar_check_column(dataset_name: str,
                             tool: LanguageTool) -> None:
    df = pd.read_csv(f"data/result/question/{dataset_name}.tsv", sep="\t")

    grammar_labels = []
    errors: Counter[str] = Counter()

    print(f"checking grammar for {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        gramatical_errors = tool.check(str(row["statement"]))

        label = "correct" if len(gramatical_errors) != 0 else "incorrect"
        grammar_labels.append(label)
        for error in gramatical_errors:
            errors[error] += 1

    df["grammar"] = grammar_labels

    error_df = pd.DataFrame(errors)
    error_df.to_csv(f"data/result/grammar/errors/{dataset_name}.tsv",
                    sep="\t", index=False)

    df.to_csv(f"data/result/grammar/{dataset_name}.tsv",
              sep="\t", index=False)


if __name__ == '__main__':
    tool = LanguageTool('en-US')

    add_grammar_check_column("test2", tool)
    add_grammar_check_column("train2", tool)
    add_grammar_check_column("val2", tool)
