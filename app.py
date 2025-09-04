import streamlit as st
import torch
import torchvision.utils as vutils
from model import Generator   # make sure model.py is in same folder

# -----------------------------
# Config (must match training)
# -----------------------------
Z_DIM = 100        # latent vector size
IMG_CHANNELS = 3   # RGB images
G_FEAT = 64        # generator feature maps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load the trained generator
# -----------------------------
@st.cache_resource
def load_generator():
    generator = Generator(z_dim=Z_DIM, img_channels=IMG_CHANNELS, gfeat=G_FEAT).to(DEVICE)
    state_dict = torch.load("generator.pth", map_location=DEVICE)
    generator.load_state_dict(state_dict)   # must match architecture
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
    # Save grid as a temporary image
    out_path = "generated.png"
    vutils.save_image(fake_images, out_path, normalize=True, nrow=min(4, num_images))
    return out_path

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎨 CIFAR-10 GAN Image Generator")
st.write("This app uses a DCGAN trained on CIFAR-10 to generate new synthetic images.")

num = st.slider("Number of images to generate:", 1, 16, 4)

if st.button("Generate"):
    img_path = generate_images(num)
    st.image(img_path, caption=f"{num} generated CIFAR-10-like images", use_column_width=True)
es", use_column_width=True)
