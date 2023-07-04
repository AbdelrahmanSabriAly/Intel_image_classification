import tensorflow as tf
import numpy as np
import streamlit as st
#import cv2
from PIL import Image,ImageOps
import os
from tensorflow.keras.models import load_model as tfk__load_model

st.set_page_config(layout="wide",page_title="Intel Image Classification")
#st.header("Intel Image Classification")
def header(url):
     st.markdown(f'<p style="background-color:#201434;color:#FF80E0;font-size:42px;border-radius:2%;font-weight:bold;">{url}</p>', unsafe_allow_html=True)
header("Intel Image Classification")
hide_st_style = """
<style>
MainMenu {visibility: hidden;}
footer{visibility: hidden;}
</style>
"""
st.markdown(hide_st_style,unsafe_allow_html=True)

def import_and_predict(image_data,mode):
    size = (256,256)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

path = os.path.dirname(__file__)
model = tfk__load_model('model.h5')
class_names = ['Buildings','Forest','Glacier','Mountain','Sea','Street']


about_tab,app_tab,contacts_tab = st.tabs(['About','App','Contact'])

# --------------------------------------- About tab ---------------------------------------------- #
about_tab.subheader("Image classification task from Kaggle to classify 6 categories:")

about_tab.write("Buildings 🏢")
about_tab.write("Forest 🌲")
about_tab.write("Glacier 🗻")
about_tab.write("Mountain ⛰️")
about_tab.write("Sea 💦")
about_tab.write("Street 🛣️")

about_tab.write("In this project, transfer learning is used with the pre-trained MobileNetV2 model")
about_tab.write("Obtained accuracy: 90.57%")
about_tab.write("You can find the link of the dataset in the following link")
about_tab.markdown("[Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)")
about_tab.write("Link of GitHub repository:")
about_tab.markdown("[GitHub Repo](https://github.com/AbdelrahmanSabriAly/Intel_image_classification.git)")


# --------------------------------------- App tab ---------------------------------------------- #
file = app_tab.file_uploader("Please upload an image of one of the six categories",type=["jpg","png","jpeg","bmp"])
if file is None:
    app_tab.text("Please upload an image file")
else:
    image = Image.open(file)
    app_tab.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    confidence = round(100*(np.max(predictions[0])),2)
    idx = np.argmax(predictions[0])
    app_tab.info(class_names[idx])
    app_tab.warning(f"Confidence: {confidence}%")


# --------------------------------------- Contact tab ---------------------------------------------- #
contacts_tab.subheader("Abdelrahman Sabri Aly")
contacts_tab.write("Email: aaly6995@gmail.com")
contacts_tab.write("Phone: +201010681318")
contacts_tab.markdown("[WhatsApp:]( https://wa.me/+201010681318)")
contacts_tab.markdown("[Linkedin](https://www.linkedin.com/in/abdelrahman-sabri)")
