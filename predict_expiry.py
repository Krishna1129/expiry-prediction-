import argparse

from expiry_predictor import predict_expiry_days
from image_model import predict_item


def main():
    parser = argparse.ArgumentParser(description="Predict produce class and expiry days")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("avg_temp_c", type=float, help="Average temperature (C)")
    parser.add_argument("humidity_pct", type=float, help="Humidity percentage")
    parser.add_argument("storage_type", help="Storage type, e.g. Room/Refrigerated/ColdStorage")
    parser.add_argument("location", help="Location, e.g. Urban/Rural/Coastal/Mountain")
    parser.add_argument("season", help="Season, e.g. Summer/Winter/Spring/Monsoon")
    args = parser.parse_args()

    item_name, item_type = predict_item(args.image_path)
    expiry_days = predict_expiry_days(
        item_name=item_name,
        item_type=item_type,
        avg_temp_c=args.avg_temp_c,
        humidity_pct=args.humidity_pct,
        storage_type=args.storage_type,
        location=args.location,
        season=args.season,
    )

    print(f"item_name: {item_name}")
    print(f"item_type: {item_type}")
    print(f"expiry_days: {expiry_days}")


if __name__ == "__main__":
    main()
