from ds_loader import load_dataset


def process_dataset(dataset_name: str) -> None:
    print(f"checking length of sentences for: {dataset_name}")
    df = load_dataset(dataset_name)

    lengths_of_sentences: list[int] = []

    for _, row in df.iterrows():
        lengths_of_sentences.append(len(row.statement))

    df["statement_length"] = lengths_of_sentences

    df["statement_length"].to_csv(f"data/result/statement_length/{dataset_name}.tsv",
                    sep="\t", index=False)
    

if __name__ == '__main__':
    process_dataset("test2")
    process_dataset("train2")
    process_dataset("val2")