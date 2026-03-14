from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("groups", None)
        return super().from_config(config)


LABELS = {
    0: "apple",
    1: "banana",
    2: "beetroot",
    3: "bell pepper",
    4: "cabbage",
    5: "capsicum",
    6: "carrot",
    7: "cauliflower",
    8: "chilli pepper",
    9: "corn",
    10: "cucumber",
    11: "eggplant",
    12: "garlic",
    13: "ginger",
    14: "grapes",
    15: "jalepeno",
    16: "kiwi",
    17: "lemon",
    18: "lettuce",
    19: "mango",
    20: "onion",
    21: "orange",
    22: "paprika",
    23: "pear",
    24: "peas",
    25: "pineapple",
    26: "pomegranate",
    27: "potato",
    28: "raddish",
    29: "soy beans",
    30: "spinach",
    31: "sweetcorn",
    32: "sweetpotato",
    33: "tomato",
    34: "turnip",
    35: "watermelon",
}

VEGETABLES = {
    "beetroot",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "corn",
    "cucumber",
    "eggplant",
    "ginger",
    "lettuce",
    "onion",
    "peas",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
}

_MODEL_PATH = Path(__file__).resolve().parent / "FV.h5"
_MODEL = tf.keras.models.load_model(
    str(_MODEL_PATH), custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D}, compile=False
)


def _predict_from_array(img_array):
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, [0])
    answer = _MODEL.predict(img_array, verbose=0)
    y_class = answer.argmax(axis=-1)
    idx = int(y_class[0])

    item_name = LABELS[idx]
    item_type = "Vegetable" if item_name in VEGETABLES else "Fruit"
    return item_name.capitalize(), item_type


def predict_item(image_input):
    if isinstance(image_input, (str, Path)):
        img = tf.keras.preprocessing.image.load_img(image_input, target_size=(224, 224, 3))
        img = tf.keras.preprocessing.image.img_to_array(img)
        return _predict_from_array(img)

    if isinstance(image_input, Image.Image):
        resized = image_input.convert("RGB").resize((224, 224))
        img = tf.keras.preprocessing.image.img_to_array(resized)
        return _predict_from_array(img)

    raise TypeError("image_input must be a file path or PIL.Image.Image")


def warmup_image_model():
    dummy = np.zeros((224, 224, 3), dtype=np.float32)
    _predict_from_array(dummy)
