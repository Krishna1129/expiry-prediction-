from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from expiry_predictor import predict_expiry_days
from image_model import predict_item


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "upload_images"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
STORAGE_TYPES = ["Room", "Refrigerated", "ColdStorage"]
LOCATIONS = ["Urban", "Rural", "Coastal", "Mountain"]
SEASONS = ["Summer", "Winter", "Spring", "Monsoon"]

app = Flask(__name__)


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

            file_name = secure_filename(image_file.filename)
            image_path = UPLOAD_DIR / file_name
            image_file.save(image_path)

            item_name, item_type = predict_item(str(image_path))
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
