from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model("deployment_28042020")


def video_youtube(
    src: str = "https://www.youtube.com/embed/B2iAodr0fOo", width="100%", height=315
):
    """An extension of the video widget
    Arguments:
        src {str} -- A youtube url like https://www.youtube.com/embed/B2iAodr0fOo
    Keyword Arguments:
        width {str} -- The width of the video (default: {"100%"})
        height {int} -- The height of the video (default: {315})
    """
    st.write(
        f'<iframe width="{width}" height="{height}" src="{src}" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
        unsafe_allow_html=True,
    )


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df["Label"][0]
    return predictions


def run():

    from PIL import Image

    with open("style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # Load images into variable.
    image = Image.open("logo.png")
    image_hospital = Image.open("hospital.jpg")
    # Looks like you can add HTML in a hacky way
    st.markdown(
        "<h1>Hacky title</h1>", unsafe_allow_html=True,
    )
    # Loads image at the top of the app (align with hack)
    st.image(image, use_column_width=False, width=200)

    # This is awesome
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch", "YouTube")
    )

    st.sidebar.info("This app is created to predict patient hospital charges")
    st.sidebar.success(
        "https://www.varonis.com/blog/kerberos-authentication-explained/.org"
    )

    st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction App")

    if add_selectbox == "Online":

        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10, max_value=50, value=10)
        children = st.selectbox("Children", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        if st.checkbox("Smoker"):
            smoker = "yes"
        else:
            smoker = "no"
        region = st.selectbox(
            "Region", ["southwest", "northwest", "northeast", "southeast"]
        )

        output = ""

        input_dict = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
        }
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = "$" + str(output)

        st.success("The output is {}".format(output))

    # Disable warning
    st.set_option("deprecation.showfileUploaderEncoding", False)
    if add_selectbox == "Batch":

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

    if add_selectbox == "YouTube":
        if st.button("Show video"):
            video_youtube()


if __name__ == "__main__":
    run()
