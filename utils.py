import random
import re
import pandas as pd
from datasets import Dataset, DatasetDict, Audio
from typing import Any, Dict, List, Optional, Union
from IPython.display import display
import json


# Paths to data folder, CSV file names
DATA_PATH = "data/"
# TRAIN_CSV = DATA_PATH + "train.csv"
TRAIN_CSV = DATA_PATH + "train_aug.csv"

DEV_CSV = DATA_PATH + "dev.csv"
TEST_CSV = DATA_PATH + "test_release.csv"


chars_to_ignore = '[y\(\“\'\’\«\—\‘\»\„\–\”]'
# From https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb#scrollTo=72737oog2F6U
def remove_special_characters(batch):
    batch["transcript"] = re.sub(chars_to_ignore, '', batch["transcript"]) + " "
    return batch


# Convert pandas data frame structure to datasets and cast audio column
# Uses the GEO dataset
def create_audio_dataset(df, DATA_PATH):
    df["audio"] = df["file"].apply(lambda x: f"{DATA_PATH}{x}")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio())
    return dataset


# Create a dataset based on the GEO dataset
def create_data_set(DATA_PATH, TRAIN_CSV, DEV_CSV, TEST_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(DEV_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_dataset = create_audio_dataset(train_df, DATA_PATH)
    val_dataset = create_audio_dataset(val_df, DATA_PATH)
    test_dataset = create_audio_dataset(test_df, DATA_PATH)
    dataset = DatasetDict({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    })
    return dataset


# From https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb#scrollTo=72737oog2F6U
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    df = pd.DataFrame(dataset[picks])
    display(df)


