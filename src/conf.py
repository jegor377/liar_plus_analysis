DATA_DIR = "./data"
PLOTS_DIR = "./plots"
CSV_DIR = "./csv"

DATASETS_BALANCE_DIR = f"{PLOTS_DIR}/balance"
DATASETS_BALANCE_DIFFS_DIR = f"{PLOTS_DIR}/balance_diffs"
DATASETS_BALANCE_MEDIAN_DIR = f"{PLOTS_DIR}/balance_median"
DATASETS_COUNTS_DIR = f"{PLOTS_DIR}/counts"
DATASETS_GRAMMAR_CHECK_PLOT_DIR = f"{PLOTS_DIR}/grammar_check"
DATASETS_QUESTIONS_COUNT_DIR = f"{PLOTS_DIR}/questions"
DATASETS_COUNT_MIN_MAX_CHARS_PLOT_DIR = f"{PLOTS_DIR}/count_min_max_chars"

DATASETS_GRAMMAR_CHECK_CSV_DIR = f"{CSV_DIR}/grammar_check"
DATASETS_QUESTIONS_COUNT_CSV_DIR = f"{CSV_DIR}/questions"
DATASETS_COUNT_MIN_MAX_CHARS_CSV_DIR = f"{CSV_DIR}/count_min_max_chars"

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
