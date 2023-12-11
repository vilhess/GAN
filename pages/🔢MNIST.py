import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
from models import Generator
from utils import load_checkpoint
from dotenv import load_dotenv
import os

load_dotenv()

DEVICE = torch.device(os.getenv('DEVICE'))

st.set_page_config(
    page_title='Deep Convolutional Generative Adversarial Network (DCGAN)')

st.markdown("<h1 style='text-align: center;'>Deep Convolutional Generative Adversarial Network (DCGAN)</h1>",
            unsafe_allow_html=True)

st.subheader("")
st.subheader("")

st.subheader("🎨 Examples of Training Images")
st.subheader("")
st.markdown("<h3 style='text-align: center;'>Displayed below are some examples of images that were used to train our model. 👇</h3>",
            unsafe_allow_html=True)
st.subheader("")
st.subheader("")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    img = Image.open("images/some_mnist_images/img0.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img5.png")
    st.image(img, use_column_width=True)
with col2:
    img = Image.open("images/some_mnist_images/img1.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img6.png")
    st.image(img, use_column_width=True)
with col3:
    img = Image.open("images/some_mnist_images/img2.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img7.png")
    st.image(img, use_column_width=True)
with col4:
    img = Image.open("images/some_mnist_images/img3.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img8.png")
    st.image(img, use_column_width=True)
with col5:
    img = Image.open("images/some_mnist_images/img4.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img9.png")
    st.image(img, use_column_width=True)

st.subheader("")
st.subheader("")

st.subheader("📚 Model Selection")
st.subheader("")
st.write("Here you can select the model you want from a particular epoch and then generate new images from it. The training was done for 50 epochs.")
st.subheader("")
model_epoch = st.slider("Choose the epoch of the model you want to use",
                        1, 50, 50, step=1)
model_epoch = model_epoch - 1

model = Generator(2).to(DEVICE)
load_checkpoint(checkpoint=torch.load(
    f"weights/checkpoint_{model_epoch}.pth.tar"), model=model)
model.eval()
st.subheader("")
st.markdown("<h2 style='text-align: center;'>🎨 Generate New Images</h2>",
            unsafe_allow_html=True)
st.subheader("")
st.subheader("")

col1, col2 = st.columns(2)

st.subheader("")

with col1:

    st.subheader('')
    st.subheader('')
    coord1 = st.slider("Choose the first coordinate of the image you want to generate",
                       float(-10), float(10), float(0), step=0.1)
    coord2 = st.slider("Choose the second coordinate of the image you want to generate",
                       float(-10), float(10), float(0), step=0.1)

with col2:
    epsilon = torch.tensor([[coord1, coord2]]).to(DEVICE)
    fake = model(epsilon).detach()
    fake = (fake + 1) / 2  # Scale the generated images to the range [0, 1]
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(fake[0][0].cpu().numpy(), cmap='gray')
    plt.axis('off')
    st.pyplot(fig)

image = Image.open("distrib.png")
st.image(image, caption="2D dimension cluster", use_column_width=True)
st.header(" ")

st.markdown("<h3 style='text-align: center;'>Using this graph, we can generate new digits by selecting coordinates of our choice.</h3>",
            unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center;'>To generate this graph, we begin by randomly generating 2D samples. Subsequently, we input these samples into the generator, which produces corresponding images. These generated images are then passed through a Convolutional Neural Network (CNN) to estimate the associated digits. Finally, we color a map based on the identified digits, creating a visual representation.</h4>",
            unsafe_allow_html=True)
