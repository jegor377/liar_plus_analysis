from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable

import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from conf import (DATASETS_QUESTIONS_COUNT_CSV_DIR,
                  DATASETS_QUESTIONS_COUNT_DIR, label_colors)
from ds_loader import load_dataset

ProcessResult = tuple[Counter[str], Counter[str], list[tuple[str, str, str]]]


def autopct_format(values: list[int]) -> Callable[[float], str]:
    def my_format(pct: float) -> str:
        total = sum(values)
        val = int(pct*total/100.0)
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def process_dataset(name: str,
                    model: AutoModelForSequenceClassification,
                    tokenizer: AutoTokenizer,
                    nlp: spacy.Language) -> ProcessResult:
    dataset = load_dataset(name)

    statements_counter: Counter[str] = Counter()
    justifications_counter: Counter[str] = Counter()
    questions = []

    print(f"counting questions in {name} dataset...")

    for _, row in tqdm(dataset.iterrows()):
        label = row.label
        statement = row.statement
        justification = row.justification

        if statement is not None:
            doc = nlp(statement)

            for sentence in doc.sents:
                s = str(sentence)
                inputs = tokenizer(s, return_tensors="pt")
                outputs = model(**inputs)
                out_class = outputs.logits.softmax(dim=1).argmax(dim=1)[0]
                statements_counter[label] += int(out_class)
                if out_class == 1:
                    questions.append((row.id, label, s))

        if justification is not None:
            doc = nlp(justification)

            for sentence in doc.sents:
                s = str(sentence)
                inputs = tokenizer(s, return_tensors="pt")
                outputs = model(**inputs)
                out_class = outputs.logits.softmax(dim=1).argmax(dim=1)[0]
                justifications_counter[label] += int(out_class)
                if out_class == 1:
                    questions.append((row.id, label, s))

    return statements_counter, justifications_counter, questions


def plot_question_count(name: str,
                        statements_counter: Counter[str],
                        justifications_counter: Counter[str]) -> None:
    print(f"plotting {name} dataset...")

    plt.cla()
    plt.clf()
    plt.close()
    fig, axis = plt.subplots(1, 2)
    axis[0].pie(statements_counter.values(),
                labels=list(statements_counter.keys()),
                autopct=autopct_format(list(statements_counter.values())),
                colors=label_colors(list(statements_counter.keys())),
                textprops={'fontsize': 8})
    axis[0].title.set_text("statements")
    axis[1].pie(justifications_counter.values(),
                labels=list(justifications_counter.keys()),
                autopct=autopct_format(list(justifications_counter.values())),
                colors=label_colors(list(justifications_counter.keys())),
                textprops={'fontsize': 8})
    axis[1].title.set_text("justifications")
    fig.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.85)
    fig.suptitle(f"questions in {name}", y=0.9)
    plt.savefig(
        f"{DATASETS_QUESTIONS_COUNT_DIR}/{name}.png", dpi=400)
    print(f'{name} dataset plotted correctly!')


def save_questions(name: str, questions: list[tuple[str, str, str]]) -> None:
    print(f'creating csv for {name} dataset...')

    csv_filepath = f"{DATASETS_QUESTIONS_COUNT_CSV_DIR}/{name}.csv"

    with open(csv_filepath, 'w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["id", "label", "question"])

        for row in tqdm(questions):
            writer.writerow(row)


def save_plot_and_csv(name: str,
                      model: AutoModelForSequenceClassification,
                      tokenizer: AutoTokenizer,
                      nlp: spacy.Language) -> None:
    (statements_counter,
     justifications_counter,
     questions) = process_dataset(name, model, tokenizer, nlp)

    save_questions(name, questions)

    plot_question_count(name, statements_counter, justifications_counter)


if __name__ == '__main__':
    print("starting question calculation...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mrsinghania/asr-question-detection")

    model = AutoModelForSequenceClassification.from_pretrained(
        "mrsinghania/asr-question-detection")

    nlp = spacy.load("en_core_web_sm")

    save_plot_and_csv("test2", model, tokenizer, nlp)
    save_plot_and_csv("train2", model, tokenizer, nlp)
    save_plot_and_csv("val2", model, tokenizer, nlp)
