import argparse

import matplotlib.pyplot as plt
import pandas as pd


def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def plot_balance(name: str) -> None:
    dataset = pd.read_csv(f'./data/{name}.tsv', header=None, sep="\t")
    dataset.columns = [
        "id",
        "json_id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state",
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
        "justification"
    ]
    label_counts = dataset.label.value_counts()
    plt.cla()
    plt.clf()
    plt.close()
    plt.title(f'{name}.tsv dataset balance')
    plt.pie(label_counts, labels=label_counts.index,
            autopct=autopct_format(label_counts))
    plt.savefig(f"./datasets_balance/{name}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Creates balance plot for dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "name",
        type=str,
        help="Dataset file name without extension in data directory")
    args = parser.parse_args()
    config = vars(args)
    name = config['name']
    plot_balance(name)
    print(f'{name} dataset plotted correctly!')
