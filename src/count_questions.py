import pandas as pd
from tqdm import tqdm
from transformers import Pipeline, pipeline


def add_question_column(dataset_name: str,
                        pipe: Pipeline) -> None:
    df = pd.read_csv(f"data/result/sentiment/{dataset_name}.tsv", sep="\t")

    questions = []

    print(f"counting questions of {dataset_name}...")

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        label = pipe(str(row["statement"]))[0]["label"]
        questions.append(label)

    df["question"] = questions

    df.to_csv(f"data/result/question/{dataset_name}.tsv",
              sep="\t",
              index=False)


if __name__ == '__main__':
    pipe = pipeline("text-classification",
                    "mrsinghania/asr-question-detection",
                    truncation=True, max_length=512)

    pipe.model.config.id2label = {0: "not_question", 1: "question"}

    add_question_column("test2", pipe)
    add_question_column("train2", pipe)
    add_question_column("val2", pipe)
