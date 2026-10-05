"""
Lungprints — Streamlit demo
Explainable CNN for Pediatric Pneumonia Classification from Chest X-Rays

Run locally:
    streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Lungprints — Pneumonia Classifier",
    page_icon="🫁",
    layout="wide",
)

DEVICE = torch.device("cpu")
IMG_SIZE = 96
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_PATH = Path(__file__).parent / "results" / "models" / "densenet121_best.pt"

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 2)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    m.to(DEVICE)
    return m, ckpt.get("epoch", None), ckpt.get("val_metrics", {})

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ------------------------------------------------------------------
# Grad-CAM (same approach as Notebook 05)
# ------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        def forward_hook(module, inp, out):
            self.activations = out
            if out.requires_grad:
                out.retain_grad()
        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, img_tensor, class_idx=1):
        self.model.zero_grad()
        self.activations = None
        logits = self.model(img_tensor)
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        acts  = self.activations
        grads = self.activations.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        probs = torch.softmax(logits, dim=1)[0].detach().numpy()
        return cam.detach().numpy(), probs

def overlay_cam(pil_img, cam, alpha=0.45, cmap="jet"):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    img_rgb = np.stack([img] * 3, axis=-1)
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8))
             .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    ) / 255.0
    heat = plt.get_cmap(cmap)(cam_resized)[..., :3]
    over = (1 - alpha) * img_rgb + alpha * heat
    return np.clip(over, 0, 1)

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🫁 Lungprints")
st.caption(
    "Explainable CNN for Pediatric Pneumonia Classification from Chest Radiographs"
)

st.warning(
    "**Research and educational use only.** This model is not a medical device "
    "and must not be used for clinical diagnosis or triage."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        - **Model:** DenseNet121 (transfer learning)
        - **Input:** pediatric chest X-ray (grayscale)
        - **Classes:** NORMAL vs. PNEUMONIA
        - **Dataset:** Chest X-Ray Images (Pneumonia), Paul Mooney / Kaggle
        - **Explainability:** Grad-CAM

        **Caveats**
        - Single-source dataset (pediatric, ages 1–5).
        - Not validated externally.
        - Grad-CAM shows attention, not clinical evidence.
        """
    )
    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
try:
    model, best_epoch, val_metrics = load_model()
    st.sidebar.success(f"Model loaded (epoch {best_epoch})")
except Exception as e:
    st.error(f"Failed to load model from `{MODEL_PATH}`.\n\n{e}")
    st.stop()

target_layer = model.features.denseblock4
gradcam = GradCAM(model, target_layer)

# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a chest X-ray (JPEG / PNG)", type=["jpg", "jpeg", "png"]
)

col_left, col_right = st.columns(2)

if uploaded is None:
    with col_left:
        st.info("Upload an image to see a prediction.")
    st.stop()

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
try:
    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("L")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)[0].numpy()

pred_idx = int(np.argmax(probs))
pred_label = "PNEUMONIA" if pred_idx == 1 else "NORMAL"
confidence = float(probs[pred_idx])

# ------------------------------------------------------------------
# Display
# ------------------------------------------------------------------
with col_left:
    st.subheader("Input X-ray")
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

with col_right:
    st.subheader("Prediction")
    if pred_label == "PNEUMONIA":
        st.error(f"**{pred_label}** — confidence {confidence:.2%}")
    else:
        st.success(f"**{pred_label}** — confidence {confidence:.2%}")

    st.markdown("**Class probabilities**")
    st.progress(float(probs[0]), text=f"NORMAL: {probs[0]:.2%}")
    st.progress(float(probs[1]), text=f"PNEUMONIA: {probs[1]:.2%}")

    if val_metrics:
        with st.expander("Model validation metrics"):
            for k, v in val_metrics.items():
                st.write(f"- **{k}**: {v:.4f}")

# ------------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------------
if show_gradcam:
    st.subheader("Grad-CAM — where the model looked")
    cam, _ = gradcam.generate(img_tensor, class_idx=pred_idx)
    overlay = overlay_cam(pil_img, cam)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(pil_img, caption="Original", use_column_width=True)
    with c2:
        st.image(cam, caption="Heatmap", use_column_width=True, clamp=True)
    with c3:
        st.image(overlay, caption="Overlay", use_column_width=True)

    st.caption(
        "Grad-CAM highlights regions that influenced the prediction. "
        "It does **not** confirm that a highlighted region is a lesion."
    )

st.divider()
st.caption("Lungprints — research prototype. Not for clinical use.")