from ds_loader import load_dataset


def process_dataset(dataset_name: str) -> None:
    print(f"checking ratio of capital letters in statements for: {dataset_name}")
    df = load_dataset(dataset_name)

    percent_of_capital_letters: list[float] = []

    for _, row in df.iterrows():
        capital_letters_count = sum(1 for c in row.statement if c.isupper())
        length_of_statement = len(row.statement)
        ratio = round(capital_letters_count/length_of_statement * 100, 2)
        percent_of_capital_letters.append(ratio)

    df["ratio_of_capital_letters"] = percent_of_capital_letters

    df["ratio_of_capital_letters"].to_csv(f"data/result/capital_letters_ratio/{dataset_name}.tsv",
                    sep="\t", index=False)
    

if __name__ == '__main__':
    process_dataset("test2")
    process_dataset("train2")
    process_dataset("val2")