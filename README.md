# Fruit and Vegetable Recognition with Expiry Prediction

This project combines two models in one Flask web app:

1. Image classifier model
: Predicts item_name and item_type (Fruit or Vegetable) from an uploaded image.
2. Expiry regression model
: Predicts expiry_days using item_name, item_type, and environment/storage inputs.

## Features

1. Upload produce image (jpg/jpeg/png).
2. Predict item name and type from the vision model.
3. Predict estimated expiry days from tabular model.
4. Single UI flow in Flask.
5. Ready for deployment on Render.

## Tech Stack

1. Python
2. TensorFlow / Keras (image model)
3. scikit-learn (expiry model)
4. Flask + Gunicorn

## Project Files

1. App.py
: Main Flask web app (UI + integrated prediction flow).
2. image_model.py
: Vision inference module.
3. expiry_predictor.py
: Expiry inference module.
4. train_expiry_model.py
: Trains and saves expiry model pipeline.
5. predict_expiry.py
: CLI entrypoint for full pipeline.
6. models/expiry_model.joblib
: Trained expiry model artifact.
7. FV.h5
: Vision model artifact.
8. render.yaml, Procfile, runtime.txt, wsgi.py
: Deployment configuration for Render.

## Input Schema for Expiry Model

Dataset file: produce_expiry_dataset.csv

Columns:

1. item_type
2. item_name
3. avg_temp_c
4. humidity_pct
5. storage_type
6. location
7. season
8. expiry_days

## Local Setup

1. Clone repository and open project folder.
2. Create and activate virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train Expiry Model

Run this once (or whenever dataset changes):

```bash
python train_expiry_model.py --data produce_expiry_dataset.csv --output models/expiry_model.joblib
```

The script prints validation MAE and RMSE and saves the full preprocessing+model pipeline.

## Run Flask App Locally

Option 1 (Windows helper script):

```powershell
./run_app.ps1
```

Option 2 (direct command):

```bash
python App.py
```

Open:

http://127.0.0.1:5000

## Run Full Pipeline via CLI

```bash
python predict_expiry.py <image_path> <avg_temp_c> <humidity_pct> <storage_type> <location> <season>
```

Example:

```bash
python predict_expiry.py upload_images/Image_1.jpg 25 60 Room Urban Summer
```

Output includes:

1. item_name
2. item_type
3. expiry_days

## Deploy on Render

This repository already includes required Render files:

1. render.yaml
2. Procfile
3. runtime.txt
4. wsgi.py

Steps:

1. Push code to GitHub.
2. Create a new Web Service in Render from this repository.
3. Render auto-uses render.yaml.
4. Wait for build and open deployed URL.

## Notes

1. Large model files increase build/start time.
2. TensorFlow on Render runs on CPU.
3. upload_images is runtime-generated and ignored in git except .gitkeep.
