import dvc.api
import pandas as pd

from ds_loader import load_dataset


def add_headers(dataset_name: str, params: dict) -> None:
    df = load_dataset(dataset_name)

    for stage in params["stages"]:
        print(f"joining {stage}/{dataset_name} dataset to main dataset...")
        new_column = pd.read_csv(
            f'data/result/{stage}/{dataset_name}.tsv', sep="\t")
        df = df.join(new_column)

    # drop useless id column
    df = df.drop("id", axis=1)

    df.to_csv(
        f"data/result/final/{dataset_name}.tsv",
        sep='\t',
        index=False)


if __name__ == '__main__':
    params = dvc.api.params_show()

    add_headers("test2", params)
    add_headers("train2", params)
    add_headers("val2", params)
