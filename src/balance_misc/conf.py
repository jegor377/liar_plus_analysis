DATASETS_BALANCE_DIR = "./plots/balance"
DATASETS_BALANCE_DIFFS_DIR = "./plots/balance_diffs"
DATASETS_BALANCE_MEDIAN_DIR = "./plots/balance_median"
DATA_DIR = "./data"


colors_dict = {
    'pants-fire': '#cd001a',
    'false': '#ef6a00',
    'barely-true': '#f2cd00',
    'half-true': '#79c300',
    'mostly-true': '#1961ae',
    'true': '#61007d'
}


def label_colors(labels: list[str]) -> list[str]:
    return [colors_dict[label] for label in labels]
