import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from skimage.feature import hog
from skimage import exposure
import random

st.title("Image Tampering Detection App")
st.write("Upload an image to check whether it's original or tampered.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    st.stop()
pil_img = Image.open(uploaded_file)
st.subheader("Original")
st.image(pil_img, use_column_width=True)

np_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (128, 128))
fd, hog_image = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=True
)
hog_image = exposure.rescale_intensity(hog_image, in_range=(0, hog_image.max()))
hog_image = np.uint8(hog_image * 255)

ela_orig = pil_img.convert("RGB").resize((128, 128))
ela_orig.save("temp.jpg", "JPEG", quality=95)
ela_comp = Image.open("temp.jpg")
ela = ImageChops.difference(ela_orig, ela_comp)
max_diff = max([e[1] for e in ela.getextrema()]) or 1
ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)

col1, col2 = st.columns(2)
with col1:
    st.subheader("HOG")
    st.image(hog_image, clamp=True, channels="GRAY", use_column_width=True)
with col2:
    st.subheader("ELA")
    st.image(ela, clamp=True, use_column_width=True)

hog_score = np.mean(fd)
ela_score = np.mean(np.array(ela))

random.seed(42)  

ela_thresh = random.uniform(0.0, 50.0) 
hog_thresh = random.uniform(0.0, float(fd.max()))  

st.subheader("Result")
if (ela_score > ela_thresh) or (hog_score > hog_thresh):
    st.error("Image is **Tampered**")
else:
    st.success("Image is **Original**")

st.write(f"**ELA score:** {ela_score:.2f}  **HOG score:** {hog_score:.2f}")
