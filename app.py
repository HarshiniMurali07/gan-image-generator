import streamlit as st
import torch
import torchvision.utils as vutils
from model import Generator   # import Generator class
import os

# -----------------------------
# Config
# -----------------------------
Z_DIM = 100        # must match training
IMG_CHANNELS = 3   # CIFAR-10 images
G_FEAT = 64        # feature size used in training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load the trained generator
# -----------------------------
@st.cache_resource  # cache so it's not reloaded every time
def load_generator():
    generator = Generator(z_dim=Z_DIM, img_channels=IMG_CHANNELS, gfeat=G_FEAT).to(DEVICE)
    generator.load_state_dict(torch.load("generator.pth", map_location=DEVICE))
    generator.eval()
    return generator

generator = load_generator()

# -----------------------------
# Helper to generate images
# -----------------------------
def generate_images(num_images=4):
    noise = torch.randn(num_images, Z_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake_images = generator(noise).cpu()
    # save grid as a temporary image
    out_path = "generated.png"
    vutils.save_image(fake_images, out_path, normalize=True, nrow=min(4, num_images))
    return out_path

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎨 GAN Image Generator (CIFAR-10)")
st.write("This app uses a DCGAN trained on CIFAR-10 to generate new synthetic images.")

num = st.slider("Number of images to generate:", 1, 16, 4)

if st.button("Generate"):
    img_path = generate_images(num)
    st.image(img_path, caption=f"{num} generated CIFAR-10-like images", use_column_width=True)
