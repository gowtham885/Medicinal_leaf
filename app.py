import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64
import google.generativeai as genai

genai.configure(api_key="AIzaSyAIewGMqAtMEtZMZjDJgEPNEwh_Q74yfGw")  # Getting the API key from .env file at the time of configuration
model = load_model('test.keras') #add final model file
class_dict = np.load("class_names.npy")

# Function for gemini response
def get_gemini_response(prompt):
    # Function to load Gemini Pro model and get responses
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


def predict(image):
    IMG_SIZE = (1, 224, 224, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        base64_img = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_img}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
contnt = "<p>Herbal medicines are preferred in both developing and developed countries as an alternative to " \
         "synthetic drugs mainly because of no side effects. Recognition of these plants by human sight will be " \
         "tedious, time-consuming, and inaccurate.</p> " \
         "<p>Applications of image processing and computer vision " \
         "techniques for the identification of the medicinal plants are very crucial as many of them are under " \
         "extinction as per the IUCN records. Hence, the digitization of useful medicinal plants is crucial " \
         "for the conservation of biodiversity.</p>"

if __name__ == '__main__':
    add_bg_from_local("Background.jpg")
    new_title = '<p style="font-family:sans-serif; color:white; font-size: 50px;">Medicinal Leaf Classification</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    contnt = '<p style="font-family:sans-serif; color:white; font-size: 20px;">Herbal medicines are preferred in both developing and developed countries as an alternative to synthetic drugs mainly because of no side effects \
    Recognition of these plants by human sight will be tedious, time-consuming, and inaccurate.</p>'
    st.markdown(contnt,unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((300, 300))
        st.image(img)
        if st.button("Predict"):
            pred = predict(img)
            name = class_dict[pred]
            prompt = "You are an expert in classification of medical plants and your task is to provide medicinal qualities of the plant like Diseases it cures and where it is most present in india in point wise. the plant name is given at the end" + name 
            result = f'<p style="font-family:sans-serif; color:Red; font-size: 16px;">The given image is {name}</p>'
            st.markdown(result, unsafe_allow_html=True)
            response = get_gemini_response(prompt)
            st.write(response)





