from PIL import Image
from flask import Flask, render_template, request

from expiry_predictor import predict_expiry_days, warmup_expiry_model
from image_model import predict_item, warmup_image_model


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
STORAGE_TYPES = ["Room", "Refrigerated", "ColdStorage"]
LOCATIONS = ["Urban", "Rural", "Coastal", "Mountain"]
SEASONS = ["Summer", "Winter", "Spring", "Monsoon"]

app = Flask(__name__)


def _warmup_models():
    warmup_image_model()
    warmup_expiry_model()


_warmup_models()


def _is_allowed_file(filename):
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "result": None,
        "error": None,
        "storage_types": STORAGE_TYPES,
        "locations": LOCATIONS,
        "seasons": SEASONS,
    }

    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file is None or image_file.filename == "":
            context["error"] = "Please upload an image file."
            return render_template("index.html", **context)

        if not _is_allowed_file(image_file.filename):
            context["error"] = "Allowed image types: jpg, jpeg, png."
            return render_template("index.html", **context)

        try:
            avg_temp_c = float(request.form.get("avg_temp_c", ""))
            humidity_pct = float(request.form.get("humidity_pct", ""))
            storage_type = request.form.get("storage_type", "")
            location = request.form.get("location", "")
            season = request.form.get("season", "")

            if storage_type not in STORAGE_TYPES:
                raise ValueError("Invalid storage type")
            if location not in LOCATIONS:
                raise ValueError("Invalid location")
            if season not in SEASONS:
                raise ValueError("Invalid season")

            uploaded_img = Image.open(image_file.stream).convert("RGB")
            item_name, item_type = predict_item(uploaded_img)
            expiry_days = predict_expiry_days(
                item_name=item_name,
                item_type=item_type,
                avg_temp_c=avg_temp_c,
                humidity_pct=humidity_pct,
                storage_type=storage_type,
                location=location,
                season=season,
            )

            context["result"] = {
                "item_name": item_name,
                "item_type": item_type,
                "expiry_days": expiry_days,
            }
        except Exception as exc:
            context["error"] = f"Prediction failed: {exc}"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
