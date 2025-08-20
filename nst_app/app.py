# -----------------------------
# app.py (Mask R-CNN AdaIN ready)
# -----------------------------
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import tensorflow_hub as hub

# Local modules
from nst_core_tf import run_gatys_tf, pil_to_tensor_01, tensor01_to_pil

st.set_page_config(page_title="Neural Style Transfer (NST) Demo", layout="centered")
st.title("🖌️🖼️ Neural Style Transfer Web App")
st.caption("Upload content + style, pick a model, and get a stylised image.")

# -----------------------------
# Sidebar: global settings
# -----------------------------
with st.sidebar:
    st.header("Settings")
    fixed_size = st.slider("Resize all images to (square)", 256, 768, 512, 64)
    model_choice = st.selectbox(
        "Model",
        ["TF-Hub (Fast)", "Gatys (Classic)", "AdaIN (requires weights)"],
        index=0
    )
    alpha = st.slider("α (content strength)", 0.0, 1.0, 0.5, 0.05)
    beta  = st.slider("β (style strength)", 0.0, 1.0, 0.8, 0.05)

    if model_choice == "AdaIN (requires weights)":
        adain_mode = st.radio("AdaIN Mode", ["Full Image", "Object Only"], index=0)
        if adain_mode == "Object Only":
            from adain_torch import COCO_CLASSES
            class_name = st.selectbox(
                "Choose object class (COCO dataset)",
                options=list(COCO_CLASSES.values()),
                index=list(COCO_CLASSES.values()).index("dog") if "dog" in COCO_CLASSES.values() else 0
            )
            selected_class_id = [k for k, v in COCO_CLASSES.items() if v == class_name][0]
        else:
            selected_class_id = None
    else:
        adain_mode = None
        selected_class_id = None

# -----------------------------
# Inputs
# -----------------------------
st.subheader("1) Provide a **Content** image")
use_camera = st.checkbox("Use webcam for content image")
content_pil = None
if use_camera:
    cam_data = st.camera_input("Take a snapshot")
    if cam_data:
        try: content_pil = Image.open(cam_data).convert("RGB")
        except UnidentifiedImageError: st.error("Cannot identify the camera image.")
else:
    content_file = st.file_uploader("Upload content image (JPG/PNG)", type=["jpg","jpeg","png"])
    if content_file:
        try: content_pil = Image.open(content_file).convert("RGB")
        except UnidentifiedImageError: st.error("Cannot identify the uploaded content image.")

st.subheader("2) Provide a **Style** image")
style_file = st.file_uploader("Upload style image (JPG/PNG)", type=["jpg","jpeg","png"])
style_pil = None
if style_file:
    try: style_pil = Image.open(style_file).convert("RGB")
    except UnidentifiedImageError: st.error("Cannot identify the uploaded style image.")

# -----------------------------
# Show previews
# -----------------------------
cols = st.columns(2)
with cols[0]:
    st.markdown("**Content**")
    if content_pil: st.image(content_pil, use_column_width=True)
with cols[1]:
    st.markdown("**Style**")
    if style_pil: st.image(style_pil, use_column_width=True)

st.markdown("---")

# -----------------------------
# TF-Hub model loader (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_tfhub_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# -----------------------------
# Helper: resize to fixed square
# -----------------------------
def resize_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.LANCZOS)

# -----------------------------
# Run button
# -----------------------------
if st.button("✨ Get Stylised Output"):
    if content_pil is None or style_pil is None:
        st.warning("Please provide both a content and a style image first.")
        st.stop()

    content_small = resize_square(content_pil, fixed_size)
    style_small   = resize_square(style_pil, fixed_size)

    # -----------------------------
    # TF-Hub
    # -----------------------------
    if model_choice == "TF-Hub (Fast)":
        c = pil_to_tensor_01(content_small)
        s = pil_to_tensor_01(style_small)
        model = load_tfhub_model()
        out = model(c, s)[0]
        out = tf.image.resize(out, (c.shape[1], c.shape[2]))
        blended = (1 - beta) * c + beta * out
        result = tensor01_to_pil(blended)
        st.success("Done with TF-Hub model.")
        st.image(result, caption="Stylised Result (TF-Hub + blend)", use_column_width=True)

    # -----------------------------
    # Gatys Classic
    # -----------------------------
    elif model_choice == "Gatys (Classic)":
        with st.spinner("Running Gatys optimisation (slow on CPU)..."):
            result = run_gatys_tf(
                content_small, style_small,
                content_weight=float(alpha),
                style_weight=float(beta),
                steps=120,
                tv_weight=1e-4
            )
        st.success("Done with Gatys.")
        st.image(result, caption="Stylised Result (Gatys)", use_column_width=True)

    # -----------------------------
    # AdaIN
    # -----------------------------
    else:
        from adain_torch import (
            adain_stylize_pil,
            adain_stylize_object_pil,
            get_segmentation_mask,
            check_adain_weights
        )
        ok, msg = check_adain_weights()
        if not ok:
            st.error(msg)
            st.stop()

        with st.spinner(f"Running AdaIN ({adain_mode})..."):
            if adain_mode == "Object Only":
                # get mask
                mask = get_segmentation_mask(content_small, class_id=selected_class_id)
                mask_img = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_img)

                # resize mask to match content_small
                mask_pil_resized = mask_pil.resize(content_small.size, Image.NEAREST)
                mask_img_resized = np.array(mask_pil_resized)
                st.image(mask_pil_resized, caption=f"Segmentation Mask ({class_name})", use_column_width=True)

                # semi-transparent overlay preview
                overlay = np.array(content_small).copy()
                red_overlay = np.zeros_like(overlay)
                red_overlay[..., 0] = 255
                alpha_overlay = 0.5
                overlay = np.where(mask_img_resized[..., None] > 127,
                                   (overlay*(1-alpha_overlay)+red_overlay*alpha_overlay).astype(np.uint8),
                                   overlay)
                overlay_pil = Image.fromarray(overlay)
                st.image(overlay_pil, caption="Mask Overlay Preview", use_column_width=True)

                # stylize only object
                result = adain_stylize_object_pil(
                    content_small, style_small,
                    alpha=float(beta), size=fixed_size,
                    class_id=selected_class_id
                )
            else:
                result = adain_stylize_pil(content_small, style_small, alpha=float(beta), size=fixed_size)

        st.success("Done with AdaIN.")
        caption = f"Stylised Result (AdaIN - {adain_mode}"
        if adain_mode == "Object Only": caption += f", class={class_name}"
        caption += ")"
        st.image(result, caption=caption, use_column_width=True)

st.markdown("---")
st.caption("Built with Streamlit • TensorFlow • TF-Hub • PyTorch AdaIN + Mask R-CNN segmentation")
