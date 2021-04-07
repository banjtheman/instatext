instatext: Train text classifiers instantly
====================================


`instatext` is a Python 3 library for processing textual data. It provides a simple API for training and predicting with text classifiers.




instatext leverages  `fasttext`, and `pandas`, for the heavy lifting

Features
--------

- Topic Extraction
- Similarity Search
- BM25 search ( word ranking search)
- Topic Search

Get it now
----------
    $ pip install instatext 

Requirements
------------

- Python  >= 3.5

Example
--------
To train a model pass in a csv with the text and labels

text - the text to classify
labels - comma seperated list of labels

```
text,labels
sample text,"sample"
bad stuff,"bad,sample"
good stuff,"good,sample"
```



```python
import instatext
import logging


def main():
    print("hello")

    instatext.train_model("test_data/example.csv", "test")
    instatext.predict("test", "Predict my good text",0.5)


if __name__ == "__main__":
    loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s |%(levelname)s: %(message)s", level=loglevel
    )
    main()
```
