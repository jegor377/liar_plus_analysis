import pandas as pd


def check_balance(filepath: str) -> None:
    dataset = pd.read_csv(filepath, sep="\t")
    print(len(dataset))


check_balance("./data/test2.tsv")
