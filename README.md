# LIAR PLUS Exploration Dataset Analysis

This is LIAR PLUS exploration dataset analysis (EDA) project. It consists of:

- data files that are delivered using DVC,
- pipeline scripts that are extracting new features, joining everything into one file,
- dataset analysis in jupyter notebooks,
- DVC pipeline configuration.

Each script of the pipeline generates its own result in a form of a column. The result is stored in the result CSV file as a single column and the last script in the pipeline joins all of the results and input files into a single output file that is later analysed. The analysis and its results of the analysis are stored in jupyter notebook files.

## Project structure

- `data` - The data directory.
    - `data/test2.tsv`, `data/val2.tsv`, `data/train2.tsv` - Preliminarily manually cleaned LIAR PLUS dataset files.
    - `data/result` - The results of the pipeline scripts. Each result stored in a seperate directory named after the script intention.
    - `data/articles` - The directory with generated articles. Articles take a long time to be generated and because of that they are seperated from the rest of the pipeline results in order to make sure that they won't be deleted by mistate.
- `src` - The directory that contains pipeline scripts.
- `analysis` - The analysis notebook files.
    - `analysis/Basic analysis.ipynb` - EDA of balance; basic statistics; column vocabulary; URLs; sentence count per state, author, party affiliation, job title and subject; sentence types; sentence sentiment; an average grammar errors. All done by Igor Santarek.
    - `analysis/EDA for new columns.ipynb` - EDA of political bias of the sentence; sentence offensiveness; gibberish level of the sentence; emotional character of the sentence; vulgar language level in the sentence. Aside from EDA, the notebook contains some instructions that fix column `curse`, previously synthesised in the pipline. It was mainly created by Mateusz Sztefek.
- `dvc.yaml` - The pipeline configuration file.
- `params.yaml` - The stages defined in the pipeline file. It's used by the `final_join.py` script.
- `env.yml` - The conda environment export file.

## How to run

Install environment:

```shell
conda create -n ML --file env.yml
conda activate ML
```

Update environment locally:

```shell
conda activate ML
conda env update --file env.yml --prune
```

Fetch dataset:

Always do it after changing a branch.

```shell
dvc pull
```

Run pipeline:

```shell
dvc repro
```

## Results

The final dataset file, created by the pipeline, is published in the Releases section [here](https://github.com/jegor377/liar_plus_analysis/releases/tag/1.0). You can also access the dataset from [Kaggle](https://www.kaggle.com/datasets/igorsantarek/liar-plus-final-dataset/data).

## Contributing

1. Create a branch with name of the change, for example:

```shell
git checkout -b name
```

2. Do changes, commit and push to origin

3. Do `dvc push`


4. Open new PR and add someone else to be a reviewer

