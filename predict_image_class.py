# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:53:32 2022

@author: Grunk
"""

import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
from tensorflow import keras

os.getcwd()

st.header("Mushroom Image Classifier")
def main():
    file_uploaded = st.file_uploader("choose the file",type=['jpg','png','jpeg'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result=predict_class(image)
        st.write(result)
        st.pyplot(figure)
        print(file_uploaded)
        print(result)

def predict_class(image):
    model = tf.keras.models.load_model('full_model.h5', custom_objects ={'KerasLayer':hub.KerasLayer})
    test_image=image.resize((224,224))
    test_image=preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image=np.expand_dims(test_image,axis=0)
    class_names=['Amanita Bisporigera',
                 'Amanita Muscaria',
                 'Boletus Edulis',
                 'Cantharellus',
                 'Russula Mariae'
                 ]
    predictions = model.predict(test_image, batch_size=1)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

if __name__ =="__main__":
    main()