import logging
import re
from typing import List
import fasttext
import pandas as pd
from datetime import datetime


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
    logging.info("N\t" + str(N))
    logging.info("P@{}\t{:.3f}".format(1, p))
    logging.info("R@{}\t{:.3f}".format(1, r))


def convert_csv_to_fast_text_doc(df: pd.DataFrame):
    """
    Purpose:
        Transform csv to fasttext format
    Args:
        df - Dataframe of the csv
    Returns:
        N/A
    """

    # TODO can we create text without having to use an array?
    text_array = []
    df.apply(lambda row: create_row_for_fast_text_doc(row, text_array), axis=1)

    # save it to a rand text file
    logging.info(text_array)

    final_text = ""
    for string in text_array:
        final_text += string

    # TODO should have a run folder each time we do train, to keep track of artifcats
    write_to_file("test.train", final_text)

# TODO make an output folder?
def train_model_from_csv(csv_location: str, model_name: str):
    """
    Purpose:
        Train a model from csv
    Args:
        csv_location - location of csv file
        model_name - name of model file
    Returns:
        N/A
    """

    # Open csv
    logging.info(f"Opening csv {csv_location}")
    df = pd.read_csv(csv_location)

    if not "text" in df and not "labels" in df:
        logging.error("CSV must have text and labels fields")
        raise ValueError("CSV must have text and labels fields")

    # convert df to fasttext format
    convert_csv_to_fast_text_doc(df)

    # Train model
    model = fasttext.train_supervised(
        input="test.train", epoch=50, wordNgrams=5, bucket=200000, dim=50, loss="ova"
    )

    print_results(*model.test("test.train", k=-1))
    # save model
    # now = str(datetime.now())
    model.save_model(model_name + ".bin")
