import streamlit as st
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import tempfile
import os
from streamlit_juxtapose import juxtapose
import pathlib
import sys
import random

sys.path.insert(0, r'../model')
import models

STREAMLIT_STATIC_PATH = (
        pathlib.Path(st.__path__[0]) / "static"
)

st.markdown("<h1 style='text-align: center;'>Depth Estiamtion for Skalp Videos</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Depth Estimation')
st.sidebar.subheader('Parameters')

app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['Predictd Depth', 'Compare Images'])

if app_mode == "Predictd Depth":

    choose_device = st.sidebar.radio("Choose your device", ("CPU", "CUDA"))
    video_file = st.sidebar.file_uploader("", type=['mp4'])

    tf = tempfile.NamedTemporaryFile(delete=False)

    if video_file:
        tf.write(video_file.read())
        vid = cv2.VideoCapture(tf.name)
        st.sidebar.video(tf.name)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output1.mp4', codec, 24, (1280, 720))

        if choose_device == "CPU":
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        depth_estimation_model = models.FCDenseNet().to(device)
        state = torch.load(
            "model.pt",
            map_location=device)
        step = state['step']
        epoch = state['epoch']
        depth_estimation_model.load_state_dict(state['model'])

        success, image = vid.read()
        while success:
            downsampled_img = cv2.resize(image, (0, 0), fx=1. / 5, fy=1. / 5)
            downsampled_img = downsampled_img[0:128, 0:256, :]
            downsampled_img = cv2.cvtColor(downsampled_img, cv2.COLOR_BGR2RGB)
            downsampled_img = Image.fromarray(downsampled_img)
            downsampled_img = transforms.ToTensor()(downsampled_img).unsqueeze_(0)
            downsampled_img = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            )(downsampled_img)

            with torch.no_grad():
                depth_estimation_model.eval()

                downsampled_img.to(device)

                predicted_depth_maps_1 = depth_estimation_model(downsampled_img)
                predicted_depth_maps_1 = predicted_depth_maps_1.squeeze(0)

                igg = predicted_depth_maps_1.permute(1, 2, 0).numpy()
                depth_display = cv2.applyColorMap(np.uint8(255 * igg / np.max(igg)), cv2.COLORMAP_JET).astype(np.uint8)
                resized = cv2.resize(depth_display, (1280, 720), interpolation=cv2.INTER_AREA)
                success, image = vid.read()
                out.write(resized)

        out.release()

        os.system(
            r"ffmpeg -y -i output1.mp4 -vcodec libx264 output_for_app.mp4")

        output_video = open('output_for_app.mp4', 'rb')
        out_bytes = output_video.read()
        st.video(out_bytes)

elif app_mode == 'Compare Images':
    choose_device = st.sidebar.radio("Choose your device", ("CPU", "CUDA"))
    image_file = st.sidebar.file_uploader("", ["jpg"])

    if image_file:

        if choose_device == "CPU":
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        image = np.array(Image.open(image_file))

        depth_estimation_model = models.FCDenseNet().to(device)
        state = torch.load(
            "model.pt",
            map_location=device)
        step = state['step']
        epoch = state['epoch']
        depth_estimation_model.load_state_dict(state['model'])

        downsampled_img = cv2.resize(image, (0, 0), fx=1. / 5, fy=1. / 5)
        downsampled_img = downsampled_img[0:128, 0:256, :]
        downsampled_img = Image.fromarray(downsampled_img)
        downsampled_img = transforms.ToTensor()(downsampled_img).unsqueeze_(0)
        downsampled_img = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )(downsampled_img)

        with torch.no_grad():
            depth_estimation_model.eval()
            downsampled_img.to(device)
            predicted_depth_maps_1 = depth_estimation_model(downsampled_img)
            predicted_depth_maps_1 = predicted_depth_maps_1.squeeze(0)

            igg = predicted_depth_maps_1.permute(1, 2, 0).numpy()
            depth_display = cv2.applyColorMap(np.uint8(255 * igg / np.max(igg)), cv2.COLORMAP_JET).astype(np.uint8)
            resized = cv2.resize(depth_display, (1280, 640), interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            resized = Image.fromarray(resized)
            last_image = image[0:640, :, :]
            last_image = Image.fromarray(last_image)

        counter_1 = random.randint(0, 1000000)
        counter_2 = counter_1 + 1

        IMG1 = f"image{counter_1}.jpg"
        IMG2 = f"image{counter_2}.jpg"

        resized.save(STREAMLIT_STATIC_PATH / IMG2)
        last_image.save(STREAMLIT_STATIC_PATH / IMG1)

        juxtapose(IMG1, IMG2)
