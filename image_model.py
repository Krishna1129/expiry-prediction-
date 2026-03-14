from pathlib import Path

import numpy as np
import tensorflow as tf


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
    str(_MODEL_PATH), custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D}
)


def predict_item(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224, 3))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])

    answer = _MODEL.predict(img)
    y_class = answer.argmax(axis=-1)
    idx = int(" ".join(str(x) for x in y_class))

    item_name = LABELS[idx]
    item_type = "Vegetable" if item_name in VEGETABLES else "Fruit"
    return item_name.capitalize(), item_type
