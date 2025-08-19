"""
TensorFlow utilities for Gatys-style optimisation and image conversions.
Kept small and friendly for Streamlit.
"""
from __future__ import annotations
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Image conversion helpers
# -----------------------------
def pil_to_tensor_01(img: Image.Image) -> tf.Tensor:
    """PIL.Image -> float32 Tensor of shape [1,H,W,3] in [0,1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    t = tf.convert_to_tensor(arr)[tf.newaxis, ...]
    return t

def tensor01_to_pil(t: tf.Tensor) -> Image.Image:
    """float32 Tensor [1,H,W,3] in [0,1] -> PIL.Image."""
    t = tf.clip_by_value(t, 0.0, 1.0)
    arr = (t[0].numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

# -----------------------------
# Gatys-style transfer (optimisation on pixels)
# -----------------------------

# Default VGG layers for style/content
_STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]
_CONTENT_LAYERS = ['block5_conv2']

def _vgg_layers(layer_names):
    """Creates a VGG19 model that returns a list of intermediate layer activations."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def _as_list(x):
    """Ensure x is a list (Keras returns a single Tensor when there's only 1 output)."""
    return x if isinstance(x, (list, tuple)) else [x]

def _gram_matrix(x):
    """Compute Gram matrix for a 4D tensor [1,H,W,C]."""
    x = tf.squeeze(x, axis=0)            # [H,W,C]
    x = tf.reshape(x, [-1, x.shape[-1]]) # [H*W, C]
    n = tf.cast(tf.shape(x)[0], tf.float32)
    gram = tf.matmul(x, x, transpose_a=True) / n
    return gram

def run_gatys_tf(content_img: Image.Image,
                 style_img: Image.Image,
                 content_weight: float = 0.5,
                 style_weight: float = 0.8,
                 steps: int = 120,
                 tv_weight: float = 1e-4) -> Image.Image:
    """
    Run the classic Gatys optimisation. Returns a PIL image.
    - content_weight ↔ α
    - style_weight   ↔ β
    Note: steps kept moderate for Streamlit CPU. Increase cautiously.
    """
    # Preprocess to VGG space [0..255] BGR with mean subtraction
    def _preprocess_vgg(pil_img):
        img = np.array(pil_img).astype(np.float32)
        img = tf.convert_to_tensor(img)[tf.newaxis, ...]
        # no-op resize (kept for clarity)
        img = tf.image.resize(img, (pil_img.height, pil_img.width), method='lanczos3')
        # RGB -> BGR and mean subtraction
        img = tf.reverse(img, axis=[-1])
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        img = img - mean
        return img

    def _deprocess_vgg(t):
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = t + mean
        x = tf.reverse(x, axis=[-1])  # BGR -> RGB
        x = tf.clip_by_value(x, 0.0, 255.0)
        arr = tf.cast(x[0], tf.uint8).numpy()
        return Image.fromarray(arr)

    # Build models
    style_extractor = _vgg_layers(_STYLE_LAYERS)
    content_extractor = _vgg_layers(_CONTENT_LAYERS)

    # Prepare tensors
    c = _preprocess_vgg(content_img)
    s = _preprocess_vgg(style_img)
    init = tf.Variable(c)  # start from content image

    # Extract targets (force list forms)
    style_targets = [_gram_matrix(t) for t in style_extractor(s)]
    content_targets = _as_list(content_extractor(c))

    opt = tf.keras.optimizers.Adam(learning_rate=0.02)

    @tf.function
    def _step(img_var):
        with tf.GradientTape() as tape:
            # Forward (force list forms)
            s_feats = style_extractor(img_var)
            c_feats = _as_list(content_extractor(img_var))

            # Style loss
            s_loss = 0.0
            for sf, st in zip(s_feats, style_targets):
                gm = _gram_matrix(sf)
                s_loss += tf.reduce_mean(tf.square(gm - st))
            s_loss = s_loss / float(len(_STYLE_LAYERS))

            # Content loss (now safe to iterate)
            diffs = [tf.reduce_mean(tf.square(cf - ct)) for cf, ct in zip(c_feats, content_targets)]
            c_loss = tf.add_n(diffs) / float(len(_CONTENT_LAYERS))

            # Total variation for smoothness
            tv = tf.image.total_variation(img_var)
            loss = style_weight * s_loss + content_weight * c_loss + tv_weight * tf.reduce_mean(tv)

        grads = tape.gradient(loss, img_var)
        return loss, grads

    for i in range(int(steps)):
        loss, grads = _step(init)
        opt.apply_gradients([(grads, init)])
        # Keep values bounded in VGG space
        init.assign(tf.clip_by_value(init, -150.0, 150.0))

    return _deprocess_vgg(init)
