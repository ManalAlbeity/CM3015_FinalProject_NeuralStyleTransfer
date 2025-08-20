# -----------------------------
# adain_torch.py (fixed version)
# -----------------------------
"""
AdaIN inference using PyTorch (backend module only).
Provides functions for full-image and object-only stylization.
Uses Mask R-CNN for object segmentation (supports all COCO objects).
Requires weights:
 - adain/vgg_normalised.pth
 - adain/decoder.pth
"""

import os
from typing import Tuple
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import sys

# -----------------------------
# Device
# -----------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Utilities
# -----------------------------
def check_adain_weights() -> Tuple[bool, str]:
    vgg_path = os.path.join("adain", "vgg_normalised.pth")
    dec_path = os.path.join("adain", "decoder.pth")
    ok = os.path.exists(vgg_path) and os.path.exists(dec_path)
    msg = "OK" if ok else (
        "Missing AdaIN weights. Place 'vgg_normalised.pth' and 'decoder.pth' inside the 'adain' folder."
    )
    return ok, msg

def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(size),
        T.ToTensor()
    ])
    return tfm(img).unsqueeze(0).to(_DEVICE)

def _to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1)
    return T.ToPILImage()(t.squeeze(0).cpu())

# -----------------------------
# Import AdaIN repo modules
# -----------------------------
if "adain" not in sys.path:
    sys.path.append("adain")

from net import decoder as _decoder, vgg as _vgg
from function import adaptive_instance_normalization as adain

# -----------------------------
# Load AdaIN models
# -----------------------------
def load_adain_models(weights_dir="adain"):
    vgg_path = os.path.join(weights_dir, "vgg_normalised.pth")
    dec_path = os.path.join(weights_dir, "decoder.pth")

    vgg = _vgg.to(_DEVICE).eval()
    decoder = _decoder.to(_DEVICE).eval()

    vgg.load_state_dict(torch.load(vgg_path, map_location=_DEVICE), strict=False)
    decoder.load_state_dict(torch.load(dec_path, map_location=_DEVICE), strict=False)

    for p in vgg.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False

    # Extract encoder layers
    try:
        encoder = vgg[:31]
    except TypeError:
        encoder = torch.nn.Sequential(*list(vgg.children())[:31])

    return encoder, decoder

# -----------------------------
# AdaIN stylization (Full Image)
# -----------------------------
@torch.no_grad()
def adain_stylize_pil(content_img: Image.Image, style_img: Image.Image,
                      alpha: float = 1.0, size: int = 512) -> Image.Image:
    ok, msg = check_adain_weights()
    if not ok: raise RuntimeError(msg)
    encoder, decoder = load_adain_models("adain")
    c = _to_tensor(content_img, size)
    s = _to_tensor(style_img, size)
    c_feats = encoder(c)
    s_feats = encoder(s)
    t = adain(c_feats, s_feats)
    t = alpha * t + (1 - alpha) * c_feats
    out = decoder(t)
    return _to_pil(out)

# -----------------------------
# COCO segmentation classes (80)
# -----------------------------
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
    8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign",
    14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse",
    20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"
}

# -----------------------------
# Mask R-CNN model for object detection
# -----------------------------
_mrcnn_model = maskrcnn_resnet50_fpn(pretrained=True).eval().to(_DEVICE)

def get_segmentation_mask(img: Image.Image, size: int = 512, class_id: int = 1, threshold: float = 0.5) -> torch.Tensor:
    """Return binary mask for given COCO class using Mask R-CNN."""
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(size),
        T.ToTensor()
    ])
    x = tfm(img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        preds = _mrcnn_model(x)[0]  # dict with 'boxes','labels','scores','masks'
        mask = torch.zeros_like(x[:, 0, :, :])
        for m, l, s in zip(preds['masks'], preds['labels'], preds['scores']):
            if l.item() == class_id and s.item() > threshold:
                mask = torch.maximum(mask, m[0])
        mask = (mask > 0.5).float()

    # Resize mask to original image size
    mask_pil = T.ToPILImage()(mask.squeeze(0).cpu())
    mask_pil = mask_pil.resize(img.size, Image.NEAREST)
    mask_resized = T.ToTensor()(mask_pil).unsqueeze(0).to(_DEVICE)
    return mask_resized

# -----------------------------
# Object-only AdaIN (with fallback + multi-class support)
# -----------------------------
@torch.no_grad()
def adain_stylize_object_pil(content_img: Image.Image, style_img: Image.Image,
                             alpha: float = 1.0, size: int = 512,
                             class_ids: list[int] | None = None) -> Image.Image:
    """
    Apply AdaIN only to objects matching the given COCO class IDs.
    If no objects are detected, fall back to full-image stylization.
    Supports multiple class IDs at once.
    """
    ok, msg = check_adain_weights()
    if not ok:
        raise RuntimeError(msg)

    encoder, decoder = load_adain_models("adain")
    c = _to_tensor(content_img, size)
    s = _to_tensor(style_img, size)

    # Aggregate masks from all selected classes
    if class_ids is None:
        return adain_stylize_pil(content_img, style_img, alpha=alpha, size=size)

    total_mask = torch.zeros_like(c[:, 0:1, :, :])  # shape [1,1,H,W]
    for cid in class_ids:
        mask = get_segmentation_mask(content_img, size=size, class_id=cid).to(_DEVICE)
        total_mask = torch.maximum(total_mask, mask)

    if total_mask.sum() == 0:
        print(f"[WARN] No objects detected for classes {class_ids}. Falling back to full image stylization.")
        return adain_stylize_pil(content_img, style_img, alpha=alpha, size=size)

    # AdaIN stylization
    c_feats = encoder(c)
    s_feats = encoder(s)
    t = adain(c_feats, s_feats)
    t = alpha * t + (1 - alpha) * c_feats
    out = decoder(t)

    # Apply only inside mask
    out = total_mask * out + (1 - total_mask) * c
    return _to_pil(out)
