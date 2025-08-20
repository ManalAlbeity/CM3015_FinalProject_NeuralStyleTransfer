"""
TensorFlow utilities for Gatys-style optimisation and image conversions.
Supports Streamlit-friendly progress updates with optional intermediate image callback.
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
_STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]
_CONTENT_LAYERS = ['block5_conv2']

def _vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def _gram_matrix(x):
    x = tf.squeeze(x, axis=0)            # [H,W,C]
    x = tf.reshape(x, [-1, x.shape[-1]]) # [H*W, C]
    n = tf.cast(tf.shape(x)[0], tf.float32)
    return tf.matmul(x, x, transpose_a=True) / n

# -----------------------------
# Run Gatys with optional callbacks
# -----------------------------
def run_gatys_tf(
    content_img: Image.Image,
    style_img: Image.Image,
    content_weight: float = 0.4,    # slightly lower, lets style dominate
    style_weight: float = 1.5,      # stronger style
    steps: int = 200,               # more steps for visible effect
    tv_weight: float = 5e-5,        # lower TV to avoid smoothing out style
    progress_callback=None,
    intermediate_callback=None,
    intermediate_every: int = 10 
) -> Image.Image:

    """
    Classic Gatys optimisation with optional Streamlit callbacks:
    - progress_callback(step, total_steps)
    - intermediate_callback(pil_image, step)
    """
    # Preprocess to VGG space
    def _preprocess_vgg(pil_img):
        img = np.array(pil_img).astype(np.float32)
        img = tf.convert_to_tensor(img)[tf.newaxis, ...]
        img = tf.image.resize(img, (pil_img.height, pil_img.width), method='lanczos3')
        img = tf.reverse(img, axis=[-1])
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        return img - mean

    # Deprocess back to PIL
    def _deprocess_vgg(t):
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
        x = t + mean
        x = tf.reverse(x, axis=[-1])
        x = tf.clip_by_value(x, 0.0, 255.0)
        arr = tf.cast(x[0], tf.uint8).numpy()
        return Image.fromarray(arr)

    # Models
    style_extractor = _vgg_layers(_STYLE_LAYERS)
    content_extractor = _vgg_layers(_CONTENT_LAYERS)

    # Prepare tensors
    c = _preprocess_vgg(content_img)
    s = _preprocess_vgg(style_img)
    init = tf.Variable(c)

    style_targets = [_gram_matrix(t) for t in style_extractor(s)]
    content_targets = _as_list(content_extractor(c))

    opt = tf.keras.optimizers.Adam(learning_rate=0.02)

    @tf.function
    def _step(img_var):
        with tf.GradientTape() as tape:
            s_feats = style_extractor(img_var)
            c_feats = _as_list(content_extractor(img_var))

            s_loss = tf.add_n([tf.reduce_mean(tf.square(_gram_matrix(sf) - st)) 
                               for sf, st in zip(s_feats, style_targets)]) / float(len(_STYLE_LAYERS))
            c_loss = tf.add_n([tf.reduce_mean(tf.square(cf - ct)) 
                               for cf, ct in zip(c_feats, content_targets)]) / float(len(_CONTENT_LAYERS))
            tv = tf.reduce_mean(tf.image.total_variation(img_var))
            loss = style_weight * s_loss + content_weight * c_loss + tv_weight * tv
        grads = tape.gradient(loss, img_var)
        return loss, grads

    # Optimization loop
    for i in range(steps):
        loss, grads = _step(init)
        opt.apply_gradients([(grads, init)])
        init.assign(tf.clip_by_value(init, -150.0, 150.0))

        # Progress callback
        if progress_callback:
            progress_callback(i + 1, steps)

        # Intermediate image callback
        if intermediate_callback and ((i + 1) % intermediate_every == 0 or i == steps - 1):
            intermediate_callback(_deprocess_vgg(init), i + 1)

    return _deprocess_vgg(init)
