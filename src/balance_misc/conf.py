DATASETS_BALANCE_DIR = "./plots/balance"
DATASETS_BALANCE_DIFFS_DIR = "./plots/balance_diffs"
DATASETS_BALANCE_MEDIAN_DIR = "./plots/balance_median"
DATA_DIR = "./data"


colors_dict = {
    'pants-fire': 'orangered',
    'false': 'coral',
    'barely-true': 'salmon',
    'half-true': 'peachpuff',
    'mostly-true': 'skyblue',
    'true': 'deepskyblue'
}


def label_colors(labels: list[str]) -> list[str]:
    return [colors_dict[label] for label in labels]
