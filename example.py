import instatext
import logging


def main():
    print("hello")

    instatext.train_model("test_data/example.csv", "test")
    instatext.predict("test", "Big crash on 15th and R st, stay away",0.0)


if __name__ == "__main__":
    loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s |%(levelname)s: %(message)s", level=loglevel
    )
    main()