import instatext
import logging


def main():

    # Train model on test_data/example.csv. call the model test, overwrite file
    instatext.train_model("test_data/example.csv", "test", True)

    # Predict with the model test, predict this phrase, threshold for labels
    instatext.predict("test", "Big crash on 15th and R st, stay away",0.5)


if __name__ == "__main__":
    loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s |%(levelname)s: %(message)s", level=loglevel
    )
    main()