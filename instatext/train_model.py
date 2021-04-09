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

# TODO have uses pass in thier own clean function?
# cleaning master function
def clean_text(text: str, bigrams: bool = False) -> str:
    text = text.lower()  # lower case
    text = re.sub("[" + my_punctuation + "]+", " ", text)  # strip punctuation
    text = re.sub("\s+", " ", text)  # remove double spacing
    # text = re.sub('([0-9]+)', '', text) # remove numbers

    # TODO do we want stop words?
    #
    # text_token_list = [
    #     word for word in text.split(" ") if word not in my_stopwords
    # ]  # remove stopwords

    # text_token_list = [word_rooter(word) if '#' not in word else word
    #                     for word in text_token_list] # apply word rooter
    # if bigrams:
    #     text_token_list = text_token_list+[text_token_list[i]+'_'+text_token_list[i+1]
    #                                         for i in range(len(text_token_list)-1)]
    # text = " ".join(text_token_list)
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


def convert_csv_to_fast_text_doc(df: pd.DataFrame, model_loc: str):
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
    df.apply(lambda row: create_row_for_fast_text_doc(row, text_array), axis=1)

    # save it to a rand text file
    logging.info(text_array)

    # should randomize training and validation set
    random.shuffle(text_array)

    train_text = ""
    valid_text = ""
    # do a classic 80/20 split
    train_len = math.ceil(len(text_array) * 0.8)

    for string in text_array[0:train_len]:
        train_text += string

    for string in text_array[train_len:]:
        valid_text += string

    # TODO should have a run folder each time we do train, to keep track of artifcats
    write_to_file(f"{model_loc}/instatext.train", train_text)
    write_to_file(f"{model_loc}/instatext.valid", valid_text)


# TODO make an output folder?
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
        epoch=50,
        wordNgrams=5,
        bucket=200000,
        dim=50,
        loss="ova",
    )

    print_results(*model.test(f"{model_loc}/instatext.valid", k=-1))
    # save model
    # now = str(datetime.now())
    model.save_model(f"{model_loc}/instatext.bin")
