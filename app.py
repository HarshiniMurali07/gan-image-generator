import streamlit as st
import torch
import torchvision.utils as vutils
from model import Generator
from io import BytesIO
from PIL import Image

# -----------------------------
# Config (must match training)
# -----------------------------
Z_DIM = 128        # latent vector size (your training used 128)
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
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator

generator = load_generator()

# -----------------------------
# Helper to generate images
# -----------------------------
def generate_images(num_images=16, nrow=8):
    noise = torch.randn(num_images, Z_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        fake_images = generator(noise).cpu()
    out_path = "generated.png"
    vutils.save_image(fake_images, out_path, normalize=True, nrow=nrow)
    return out_path, fake_images

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CIFAR-10 GAN Generator", page_icon="🎨", layout="wide")

st.title("🎨 CIFAR-10 GAN Image Generator")
st.markdown(
    """
    This app uses a **DCGAN** trained on **CIFAR-10** to generate synthetic images.  
    Adjust the controls in the sidebar and click **Generate** to see new results!
    """
)

# Sidebar controls
st.sidebar.header("⚙️ Controls")
num = st.sidebar.slider("Number of images", 1, 64, 16, step=1)
nrow = st.sidebar.slider("Images per row", 1, 16, 8, step=1)
generate = st.sidebar.button("🚀 Generate Images")

if generate:
    img_path, fake_images = generate_images(num, nrow)

    # Show image grid
    st.subheader("🖼️ Generated Image Grid")
    st.image(img_path, caption=f"{num} generated CIFAR-10-like images", use_column_width=True)

    # Option to download
    with open(img_path, "rb") as file:
        btn = st.download_button(
            label="📥 Download Image Grid",
            data=file,
            file_name="generated.png",
            mime="image/png"
        )

    # Expandable gallery view
    st.subheader("🔎 Explore Individual Images")
    cols = st.columns(8)
    for i, img_tensor in enumerate(fake_images):
        pil_img = transforms.ToPILImage()(0.5 * (img_tensor + 1))  # convert [-1,1] to [0,1]
        with cols[i % 8]:
            st.image(pil_img, use_column_width=True)

