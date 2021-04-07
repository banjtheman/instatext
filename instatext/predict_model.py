from typing import Any, Dict, Tuple
import fasttext


def format_predictions(predictions: Tuple) -> Dict[str, Any]:
    """
    Purpose:
        Format predictions to be readable
    Args:
        predictions: Tuple of predictions
    Returns:
        json_resp - A dictonary with the label and the confidence
    """
    len_tuple = len(predictions[1])

    json_obj = {}
    counter = 0
    while counter < len_tuple:
        label = predictions[0][counter].replace("__label__", "")
        pred = predictions[1][counter]
        json_obj[label] = pred

        counter += 1

    return json_obj


def predict(model_name: str, text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Purpose:
        Predict text using a trained model
    Args:
        model_name - name of model file
        text - to predict
        threshold - confidence of model
    Returns:
        json_resp - A dictonary with the label and the confidence
    """
    model = fasttext.load_model(model_name + ".bin")
    predictions = model.predict(text, k=-1, threshold=threshold)
    json_resp = format_predictions(predictions)

    print(json_resp)
    return json_resp
