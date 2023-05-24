import glob
import streamlit as st
from PIL import Image
import torch
import cv2
import os
import time


st.set_page_config(layout="wide")

config_model_path = 'models/antenna_yolov5l.pt'

model = None
confidence = 0.25

def image_input():
    img_file = None

    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if img_bytes:
        img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Prediction")

def video_input():
    vid_file = None
    vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
    if vid_bytes:
        vid_file = 'data/uploaded_data/upload.'+vid_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, is Video Stream Ended ?")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1/(curr_time-prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            
        cap.release()

def infer_image(img, size=None):
    model.conf = confidence
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_
    

def main():
    global model, confidence, config_model_path

    st.title("Drone Maintenance System")

    st.sidebar.title("Settings")

    if not os.path.isfile(config_model_path):
        st.warning("Model file is not Available!", icon="⚠️")
    else:
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device",["cpu", "cuda"], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device",["cpu", "cuda"], disabled=True, index=0)

        model = load_model(config_model_path, device_option)

        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=0.45)

        st.sidebar.markdown("---")

        input_option = st.sidebar.radio("Select input type: ", ['image', 'video'])

        if input_option == 'image':
            image_input()
        else:
            video_input()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass