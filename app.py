import streamlit as st
import os
import pandas as pd
from PIL import Image
import numpy as np
import tensorflow
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm


feature_list = np.array(pickle.load(open('features.pkl','rb')))
filename = pickle.load(open('filename.pkl','rb'))


model = load_model('fashion_model.h5')

st.title('Fashion Recommender System')


def save_uploaded_file(upload_file):
    try:
        with open(os.path.join('upload',upload_file.name),'wb') as f:
            f.write(upload_file.getbuffer())
        return 1
    except:
        return 0        
def feature_extraction(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    exp_img_arr = np.expand_dims(img_array,axis=0)
    preprocess_img = preprocess_input(exp_img_arr)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result/norm(result)
    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)

    distance,indices=neighbors.kneighbors([features])
    return indices




upload_file = st.file_uploader("Choose an image")
if upload_file is not None:
    if save_uploaded_file(upload_file):
        display_img =Image.open(upload_file)
        st.image(display_img)
        feature = feature_extraction(os.path.join("upload",upload_file.name),model)

        indices = recommend(feature,feature_list)
        print(indices)
        col1,col2,col3,col4,col5=st.columns(5)

        with col1:
            st.image(filename[indices[0][0]])
        with col2:
            st.image(filename[indices[0][1]])
        with col3:
            st.image(filename[indices[0][2]])
        with col4:
            st.image(filename[indices[0][3]])
        with col5:
            st.image(filename[indices[0][4]])                




    else:
        st.header('Some Error Occurred in file upload')    
