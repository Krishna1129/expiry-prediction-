import streamlit as st
from PIL import Image
import os
from expiry_predictor import predict_expiry_days
from image_model import predict_item


def run():
    st.title("Fruits🍍-Vegetable🍅 Classification")

    avg_temp_c = st.number_input("Average Temperature (C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)
    humidity_pct = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
    storage_type = st.selectbox("Storage Type", ["Room", "Refrigerated", "ColdStorage"])
    location = st.selectbox("Location", ["Urban", "Rural", "Coastal", "Mountain"])
    season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Monsoon"])

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        os.makedirs('./upload_images', exist_ok=True)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            item_name, item_type = predict_item(save_image_path)
            expiry_days = predict_expiry_days(
                item_name=item_name,
                item_type=item_type,
                avg_temp_c=avg_temp_c,
                humidity_pct=humidity_pct,
                storage_type=storage_type,
                location=location,
                season=season,
            )
            st.info('**Category : ' + item_type + '**')
            st.success("**Predicted : " + item_name + '**')
            st.warning('**Estimated Expiry : ' + str(expiry_days) + ' day(s)**')


run()
