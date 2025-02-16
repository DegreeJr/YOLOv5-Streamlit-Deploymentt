import streamlit as st
import torch
import cv2
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
from video_predict import runVideo

# Configurations
CFG_MODEL_PATH = "models/yourModel.pt"
CFG_ENABLE_URL_DOWNLOAD = True
CFG_ENABLE_VIDEO_PREDICTION = True

if CFG_ENABLE_URL_DOWNLOAD:
    url = "https://archive.org/download/yoloTrained/yoloTrained.pt"

# Function to handle real-time camera input
def realtimeCamera(model):
    st.title("Real-time Object Detection")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break
        
        # Convert frame to image
        imgpath = "temp_frame.jpg"
        cv2.imwrite(imgpath, frame)
        
        # Run prediction
        pred = model(imgpath)
        pred.render()
        
        # Convert to displayable format
        for im in pred.ims:
            img = Image.fromarray(im)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            stframe.image(img_cv, channels="BGR", use_column_width=True)
    
    cap.release()

# Function to handle image input
def imageInput(model, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width=True)
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())
            with st.spinner(text="Predicting..."):
                pred = model(imgpath)
                pred.render()
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width=True)

# Function to handle video input
def videoInput(model, src):
    if src == 'Upload your own data.':
        uploaded_video = st.file_uploader("Upload A Video", type=['mp4', 'mpeg', 'mov'])
        pred_view = st.empty()
        if uploaded_video is not None:
            ts = datetime.timestamp(datetime.now())
            uploaded_video_path = os.path.join('data/uploads', str(ts) + uploaded_video.name)
            with open(uploaded_video_path, mode='wb') as f:
                f.write(uploaded_video.read())
            st.video(uploaded_video_path)
            st.write("Uploaded Video")
            submit = st.button("Run Prediction")
            if submit:
                runVideo(model, uploaded_video_path, pred_view, st.empty())

# Main function
def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        downloadModel()
    elif not os.path.exists(CFG_MODEL_PATH):
        st.error('Model not found, please enable URL download.', icon="⚠️")

    st.sidebar.title('⚙️ Options')
    datasrc = st.sidebar.radio("Select input source.", ['From example data.', 'Upload your own data.', 'Real-time Camera'])
    option = st.sidebar.radio("Select input type.", ['Image', 'Video', 'Real-time Camera'])
    st.header('📦 YOLOv5 Streamlit Deployment Example')

    if option == "Image":
        imageInput(loadmodel(), datasrc)
    elif option == "Video":
        videoInput(loadmodel(), datasrc)
    elif option == "Real-time Camera":
        realtimeCamera(loadmodel())

@st.cache_resource
def downloadModel():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out="models/")

@st.cache_resource
def loadmodel():
    if CFG_ENABLE_URL_DOWNLOAD:
        model_path = f"models/{url.split('/')[-1]}"
    else:
        model_path = CFG_MODEL_PATH
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

if __name__ == '__main__':
    main()
