# -----------------------------
# app.py (Mask R-CNN AdaIN + Improved Gatys with multi-object & fallback)
# -----------------------------
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import tensorflow_hub as hub
import io

# Local modules
from nst_core_tf import run_gatys_tf, pil_to_tensor_01, tensor01_to_pil

st.set_page_config(page_title="Neural Style Transfer (NST) Web App", layout="centered")
st.title("🖌️ Neural Style Transfer Web App")
st.caption("Upload content + style, pick a model, and get a stylised image.")

# -----------------------------
# Helper: convert PIL to BytesIO for download
# -----------------------------
def pil_to_bytes(pil_img, format="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    buf.seek(0)
    return buf

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

    if model_choice == "Gatys (Classic)":
        st.subheader("Gatys Style Options")
        alpha = st.slider("α (content strength)", 0.0, 1.0, 0.4, 0.05)
        beta  = st.slider("β (style strength)", 0.0, 2.0, 1.5, 0.05)
        steps = st.slider("Gatys Steps", 50, 400, 200, 10)
    else:
        # For TF-Hub and AdaIN we use α/β as blend / style intensity knobs
        alpha = st.slider("α (content strength)", 0.0, 1.0, 0.5, 0.05)
        beta  = st.slider("β (style strength)",   0.0, 1.0, 0.8, 0.05)
        steps = 120  # unused outside Gatys

    # AdaIN-specific controls
    if model_choice == "AdaIN (requires weights)":
        adain_mode = st.radio("AdaIN Mode", ["Full Image", "Object Only"], index=0)

        if adain_mode == "Object Only":
            # Import here so users not using AdaIN don't need PyTorch installed
            from adain_torch import COCO_CLASSES
            # Multi-select classes
            class_names = st.multiselect(
                "Choose object class(es) (COCO dataset)",
                options=list(COCO_CLASSES.values()),
                default=["dog"] if "dog" in COCO_CLASSES.values() else []
            )
            selected_class_ids = [k for k, v in COCO_CLASSES.items() if v in class_names]

            # Detection controls
            conf_thresh = st.slider("Mask R-CNN confidence threshold", 0.30, 0.95, 0.50, 0.05)
            fallback_full = st.checkbox(
                "Fallback to full-image stylization if no objects found",
                value=True
            )
        else:
            selected_class_ids = None
            conf_thresh = 0.5
            fallback_full = True
    else:
        adain_mode = None
        selected_class_ids = None
        conf_thresh = 0.5
        fallback_full = True

    st.caption(
        "Tip: TF-Hub blends output with content by β. Gatys uses α/β as loss weights. "
        "AdaIN uses β as style intensity."
    )

# -----------------------------
# Inputs
# -----------------------------
st.subheader("1) Provide a **Content** image")
use_camera = st.checkbox("Use webcam for content image")
content_pil = None
if use_camera:
    cam_data = st.camera_input("Take a snapshot")
    if cam_data:
        try:
            content_pil = Image.open(cam_data).convert("RGB")
        except UnidentifiedImageError:
            st.error("Cannot identify the camera image.")
else:
    content_file = st.file_uploader("Upload content image (JPG/PNG)", type=["jpg","jpeg","png"])
    if content_file:
        try:
            content_pil = Image.open(content_file).convert("RGB")
        except UnidentifiedImageError:
            st.error("Cannot identify the uploaded content image.")

st.subheader("2) Provide a **Style** image")
style_file = st.file_uploader("Upload style image (JPG/PNG)", type=["jpg","jpeg","png"])
style_pil = None
if style_file:
    try:
        style_pil = Image.open(style_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("Cannot identify the uploaded style image.")

# -----------------------------
# Show previews
# -----------------------------
cols = st.columns(2)
with cols[0]:
    st.markdown("**Content**")
    if content_pil:
        st.image(content_pil, use_container_width=True)
with cols[1]:
    st.markdown("**Style**")
    if style_pil:
        st.image(style_pil, use_container_width=True)

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
if st.button("✨ Get Stylised Output", type="primary"):
    if content_pil is None or style_pil is None:
        st.warning("Please provide both a content and a style image first.")
        st.stop()

    # Always resize to the fixed square for computation
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
        st.image(result, caption="Stylised Result (TF-Hub + blend)", use_container_width=True)
        st.download_button("⬇️ Download Stylised Image", pil_to_bytes(result), "stylised_output.png", "image/png")

    # -----------------------------
    # Gatys Classic (with progress)
    # -----------------------------
    elif model_choice == "Gatys (Classic)":
        st.subheader("Gatys Style Transfer (may be slow on CPU)")
        progress_bar = st.progress(0)
        step_text = st.empty()
        image_slot = st.empty()

        def update_progress(step, total_steps):
            progress_bar.progress(step / total_steps)
            step_text.text(f"Step {step}/{total_steps}")

        def show_intermediate(pil_img, step):
            image_slot.image(pil_img, caption=f"Step {step}", use_container_width=True)

        with st.spinner("Running Gatys style transfer, please wait..."):
            result = run_gatys_tf(
                content_small, style_small,
                content_weight=float(alpha),
                style_weight=float(beta),
                steps=steps,
                tv_weight=5e-5,
                progress_callback=update_progress,
                intermediate_callback=show_intermediate,
                intermediate_every=10
            )

        st.success("Done with Gatys.")
        st.image(result, caption="Stylised Result (Gatys)", use_container_width=True)
        st.download_button("⬇️ Download Stylised Image", pil_to_bytes(result), "stylised_output_gatys.png", "image/png")

    # -----------------------------
    # AdaIN (PyTorch)
    # -----------------------------
    else:
        from adain_torch import (
            adain_stylize_pil,
            adain_stylize_object_pil,
            get_segmentation_mask,
            check_adain_weights,
            COCO_CLASSES
        )
        ok, msg = check_adain_weights()
        if not ok:
            st.error(msg)
            st.stop()

        with st.spinner(f"Running AdaIN ({adain_mode})..."):
            if adain_mode == "Object Only":
                if not selected_class_ids:
                    st.warning("Please choose at least one object class.")
                    st.stop()

                # Build a combined mask for preview
                combined_mask = None
                for cid in selected_class_ids:
                    m = get_segmentation_mask(content_small, class_id=cid, size=fixed_size, threshold=conf_thresh)
                    m_np = m.squeeze().detach().cpu().numpy()  # [H,W] in {0,1}
                    combined_mask = m_np if combined_mask is None else np.maximum(combined_mask, m_np)

                # Show mask + overlay previews
                mask_vis = (combined_mask * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_vis)
                st.image(mask_pil, caption=f"Segmentation Mask ({', '.join([COCO_CLASSES[c] for c in selected_class_ids])})", use_container_width=True)

                overlay = np.array(content_small).copy()
                red = np.zeros_like(overlay); red[..., 0] = 255
                alpha_overlay = 0.5
                overlay = np.where(
                    mask_vis[..., None] > 127,
                    (overlay * (1 - alpha_overlay) + red * alpha_overlay).astype(np.uint8),
                    overlay
                )
                st.image(Image.fromarray(overlay), caption="Mask Overlay Preview", use_container_width=True)

                # If empty mask and fallback selected → full image stylization
                if combined_mask.sum() == 0 and fallback_full:
                    result = adain_stylize_pil(content_small, style_small, alpha=float(beta), size=fixed_size)
                else:
                    # Call the new multi-class API; keep backward-compat if user still has old function
                    try:
                        result = adain_stylize_object_pil(
                            content_small, style_small,
                            alpha=float(beta), size=fixed_size,
                            class_ids=selected_class_ids
                        )
                    except TypeError:
                        # Old signature: pick the class with the largest mask area
                        areas = []
                        for cid in selected_class_ids:
                            m = get_segmentation_mask(content_small, class_id=cid, size=fixed_size, threshold=conf_thresh)
                            areas.append((cid, float(m.sum().item())))
                        best_cid = max(areas, key=lambda x: x[1])[0]
                        result = adain_stylize_object_pil(
                            content_small, style_small,
                            alpha=float(beta), size=fixed_size,
                            class_id=best_cid
                        )
            else:
                # Full-image AdaIN
                result = adain_stylize_pil(content_small, style_small, alpha=float(beta), size=fixed_size)

        # Show + download
        if adain_mode == "Object Only":
            chosen = ", ".join([COCO_CLASSES[c] for c in selected_class_ids])
            caption = f"Stylised Result (AdaIN – Object Only: {chosen})"
        else:
            caption = "Stylised Result (AdaIN – Full Image)"

        st.success("Done with AdaIN.")
        st.image(result, caption=caption, use_container_width=True)
        st.download_button("⬇️ Download Stylised Image", pil_to_bytes(result), "stylised_output.png", "image/png")

st.markdown("---")
st.caption("Built with Streamlit • TensorFlow • TF-Hub • PyTorch AdaIN + Mask R-CNN segmentation")
st.caption("Final Deliverable for CM3015 Template: Neural Style Transfer • August 2025")
