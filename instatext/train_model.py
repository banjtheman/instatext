import math
import logging
import os
import re
from typing import List
import random

import fasttext
import pandas as pd

# from datetime import datetime


my_punctuation = "#!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@“…ə"

# cleaning master function
def clean_text(text: str, bigrams: bool = False) -> str:
    text = text.lower()  # lower case
    text = re.sub("[" + my_punctuation + "]+", " ", text)  # strip punctuation
    text = re.sub("\s+", " ", text)  # remove double spacing
    # text = re.sub('([0-9]+)', '', text) # remove numbers
    return text


def write_to_file(file_path: str, file_text: str) -> bool:
    """
    Purpose:
        Write text from a file
    Args/Requests:
         file_path: file path
         file_text: Text of file
    Return:
        Status: True if appened, False if failed
    """

    try:
        with open(file_path, "w") as myfile:
            myfile.write(file_text)
            return True

    except Exception as error:
        logging.error(error)
        return False


def create_row_for_fast_text_doc(row: pd.Series, text_array: List):
    """
    Purpose:
        add cleaned text to an array
    Args:
        row - PD row
        text_array - array for text
    Returns:
        N/A
    """
    text = ""
    # get labels
    labels = row["labels"].split(",")
    logging.info(labels)

    for label in labels:
        text += "__label__" + label + " "

    text += clean_text(row["text"]) + "\n"
    logging.info(text)
    text_array.append(text)


def create_row_for_fast_text_doc_custom(
    row: pd.Series, text_array: List, cleaning_function
):
    """
    Purpose:
        add cleaned text to an array
    Args:
        row - PD row
        text_array - array for text
    Returns:
        N/A
    """
    text = ""
    # get labels
    labels = row["labels"].split(",")
    logging.debug(labels)

    for label in labels:
        text += "__label__" + label + " "

    text += cleaning_function(row["text"]) + "\n"
    logging.debug(text)
    text_array.append(text)


def print_results(N: int, p: float, r: float):
    """
    Purpose:
        Print training results
    Args:
        N - number of sentences
        p - precision
        r - recall
    Returns:
        N/A
    """
    logging.info("Number tested\t" + str(N))
    logging.info("Precision{}\t{:.3f}".format(1, p))
    logging.info("Recall{}\t{:.3f}".format(1, r))


def convert_csv_to_fast_text_doc(
    df: pd.DataFrame, model_loc: str, cleaning_function=None
):
    """
    Purpose:
        Transform csv to fasttext format
    Args:
        model_loc: model location
        df - Dataframe of the csv
    Returns:
        N/A
    """

    # TODO can we create text without having to use an array?
    text_array = []
    if not cleaning_function is None:
        df.apply(
            lambda row: create_row_for_fast_text_doc_custom(
                row, text_array, cleaning_function
            ),
            axis=1,
        )
    else:
        df.apply(lambda row: create_row_for_fast_text_doc(row, text_array), axis=1)

    # should randomize training and validation set
    random.shuffle(text_array)
    logging.info(f"text array size: {len(text_array)}")

    train_text = ""
    valid_text = ""
    # do a classic 80/20 split
    train_len = math.ceil(len(text_array) * 0.8)

    logging.info(f"train len size: {train_len}")

    for string in text_array[:train_len]:
        train_text += string

    for string in text_array[train_len:]:
        valid_text += string

    # TODO should have a run folder each time we do train, to keep track of artifcats
    write_to_file(f"{model_loc}/instatext.train", train_text)
    write_to_file(f"{model_loc}/instatext.valid", valid_text)


def train_model_from_csv(csv_location: str, model_name: str, overwrite: bool = False):
    """
    Purpose:
        Train a model from csv
    Args:
        csv_location - location of csv file
        model_name - name of model output folder
        overwrite - overwrite existing file
    Returns:
        N/A
    """

    # Open csv
    logging.info(f"Opening csv {csv_location}")
    df = pd.read_csv(csv_location)

    if not "text" in df and not "labels" in df:
        logging.error("CSV must have text and labels fields")
        raise ValueError("CSV must have text and labels fields")

        # Create model output location

    model_loc = f"instatext_model_{model_name}"

    if os.path.exists(model_loc) and not overwrite:
        raise OSError(f"Model {model_name} exists at {model_loc}")
    try:
        os.makedirs(model_loc, exist_ok=True)
    except Exception as error:
        raise OSError(error)

    # convert df to fasttext format
    convert_csv_to_fast_text_doc(df, model_loc)

    # Train model
    # TODO do we want people to specify model params?
    # if they knew what params they wanted..., they might as well use fasttext
    model = fasttext.train_supervised(
        input=f"{model_loc}/instatext.train",
        epoch=500,
        wordNgrams=5,
        bucket=200000,
        dim=50,
        loss="ova",
    )

    print_results(*model.test(f"{model_loc}/instatext.valid", k=-1))
    # save model
    # now = str(datetime.now())
    model.save_model(f"{model_loc}/instatext.bin")


def train_custom_model_from_csv(
    csv_location: str,
    model_name: str,
    overwrite: bool = False,
    model_function=None,
    cleaning_function=None,
):
    """
    Purpose:
        Train a model from csv
    Args:
        csv_location - location of csv file
        model_name - name of model output folder
        overwrite - overwrite existing file
        model_function - custom fasttext model
        cleaning_function - custom cleantext function
    Returns:
        N/A
    """

    # Open csv
    logging.info(f"Opening csv {csv_location}")
    df = pd.read_csv(csv_location)

    if not "text" in df and not "labels" in df:
        logging.error("CSV must have text and labels fields")
        raise ValueError("CSV must have text and labels fields")

        # Create model output location

    model_loc = f"instatext_model_{model_name}"

    if os.path.exists(model_loc) and not overwrite:
        raise OSError(f"Model {model_name} exists at {model_loc}")
    try:
        os.makedirs(model_loc, exist_ok=True)
    except Exception as error:
        raise OSError(error)

    # convert df to fasttext format
    convert_csv_to_fast_text_doc(df, model_loc, cleaning_function)

    # Train model
    # TODO do we want people to specify model params?
    # if they knew what params they wanted..., they might as well use fasttext

    model = model_function(f"{model_loc}/instatext.train")

    print_results(*model.test(f"{model_loc}/instatext.valid", k=-1))
    # save model
    # now = str(datetime.now())
    model.save_model(f"{model_loc}/instatext.bin")