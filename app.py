import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
from video_predict import runVideo

# Konfigurasi Model
CFG_MODEL_PATH = "models/best.pt"
CFG_ENABLE_VIDEO_PREDICTION = True

def load_model():
    return torch.hub.load("ultralytics/yolov5", "custom", path=CFG_MODEL_PATH, force_reload=True)

model = load_model()

def detect_objects(img):
    results = model([img])
    detected_img = img.copy()
    
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(detected_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return detected_img

def imageInput():
    image_file = st.file_uploader("Upload gambar burung", type=["png", "jpeg", "jpg"])
    col1, col2 = st.columns(2)
    if image_file is not None:
        img = Image.open(image_file)
        with col1:
            st.image(img, caption='Gambar yang diunggah', use_column_width=True)
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
        outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
        with open(imgpath, mode="wb") as f:
            f.write(image_file.getbuffer())
        with st.spinner(text="Memprediksi..."):
            detected_image = detect_objects(np.array(img))
            cv2.imwrite(outputpath, detected_image)
        img_ = Image.open(outputpath)
        with col2:
            st.image(img_, caption='Hasil Prediksi', use_column_width=True)

def videoInput():
    uploaded_video = st.file_uploader("Upload video burung", type=['mp4', 'mpeg', 'mov'])
    pred_view = st.empty()
    if uploaded_video is not None:
        ts = datetime.timestamp(datetime.now())
        uploaded_video_path = os.path.join('data/uploads', str(ts) + uploaded_video.name)
        with open(uploaded_video_path, mode='wb') as f:
            f.write(uploaded_video.read())
        st.video(uploaded_video_path)
        st.write("Video yang diunggah")
        submit = st.button("Jalankan Prediksi")
        if submit:
            runVideo(model, uploaded_video_path, pred_view, st.empty())

def main():
    st.title("Deteksi Burung dengan YOLOv5")
    option = st.radio("Pilih input:", ['Gambar', 'Video'])
    if option == "Gambar":
        imageInput()
    elif option == "Video":
        videoInput()

if __name__ == '__main__':
    main()
