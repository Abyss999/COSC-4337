import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL1_PATH = os.path.join(_DIR, "..", "Models", "Model2.pth")
MODEL2_PATH = os.path.join(_DIR, "..", "Models", "Model1.pth")

# class labels 
CLASSES = ["Bacterial", "Normal", "Viral"]
PNEUMONIA_CLASSES = {"Bacterial", "Viral"}

# pre processing 
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# load models 
@st.cache_resource
def load_model1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 3),
    )
    model = model.to(device)
    state_dict = torch.load(MODEL1_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


@st.cache_resource
def load_model2():
    if not os.path.exists(MODEL2_PATH):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, 3),
    )
    model = model.to(device)
    state_dict = torch.load(MODEL2_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

# inference 
def run_inference(model, device, image: Image.Image):
    """Returns (predicted_label, probabilities_list) for a PIL image."""
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_idx = probs.index(max(probs))
    return CLASSES[pred_idx], probs

# resolve conflict 
def resolve_conflict(m1_label, m1_probs, m2_label, m2_probs):
    """
    Called when Model 1 and Model 2 disagree on pneumonia type.
    Picks the model with higher confidence in its top prediction.
    Both models share the same CLASSES list (Bacterial / Normal / Viral).
    """
    m1_confidence = max(m1_probs[i] for i, c in enumerate(CLASSES) if c in PNEUMONIA_CLASSES)
    m2_confidence = max(m2_probs[i] for i, c in enumerate(CLASSES) if c in PNEUMONIA_CLASSES)

    if m1_confidence >= m2_confidence:
        return m1_label, "Model 1", m1_confidence
    else:
        return m2_label, "Model 2", m2_confidence

# chart 
def prob_bar_chart(labels, probs, title, highlight_idx=None):
    colors = []
    for i in range(len(labels)):
        if i == highlight_idx:
            colors.append("#EF553B")
        else:
            colors.append("#636EFA")

    fig = go.Figure(go.Bar(
        x=labels,
        y=[p * 100 for p in probs],
        marker_color=colors,
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(title="Probability (%)", range=[0, 110]),
        xaxis=dict(title="Class"),
        height=350,
        margin=dict(t=50, b=30),
    )
    return fig

# main ui 
def main():
    st.set_page_config(page_title="Pneumonia X-Ray Classifier", layout="wide")
    st.title("Pneumonia X-Ray Classifier")
    st.caption("DenseNet121 — Bacterial / Normal / Viral")

    uploaded = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Upload an X-ray image to get started.")
        return

    image = Image.open(uploaded).convert("RGB")

    col_img, col_results = st.columns([1, 2])

    with col_img:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col_results:
        # model 1 
        st.subheader("Model 1 — Primary Classifier")
        with st.spinner("Running Model 1..."):
            try:
                model1, device1 = load_model1()
                m1_label, m1_probs = run_inference(model1, device1, image)
            except Exception as e:
                st.error(f"Model 1 failed to load or run: {e}")
                return

        m1_pred_idx = CLASSES.index(m1_label)
        st.plotly_chart(
            prob_bar_chart(CLASSES, m1_probs, "Model 1 — Class Probabilities", highlight_idx=m1_pred_idx),
            use_container_width=True,
        )

        if m1_label == "Normal":
            st.success("Result: **NORMAL** — no pneumonia detected.")
        else:
            st.error(f"Result: **PNEUMONIA** — type: **{m1_label}**")

    # model 2 (conditional)
    if m1_label in PNEUMONIA_CLASSES:
        st.divider()
        st.subheader("Model 2 — Secondary Classifier")

        model2, device2 = load_model2()

        if model2 is None:
            st.warning(
                "Model 2 is not yet available (`Models/Model3.pth` not found). "
                "Add the model file to enable secondary classification."
            )
        else:
            with st.spinner("Running Model 2..."):
                try:
                    m2_label, m2_probs = run_inference(model2, device2, image)
                except Exception as e:
                    st.error(f"Model 2 failed to run: {e}")
                    return

            m2_pred_idx = CLASSES.index(m2_label)
            st.plotly_chart(
                prob_bar_chart(CLASSES, m2_probs, "Model 2 — Class Probabilities", highlight_idx=m2_pred_idx),
                use_container_width=True,
            )

            # verdict 
            st.divider()
            st.subheader("Final Verdict")

            if m1_label == m2_label:
                st.success(f"Both models agree: **{m1_label} Pneumonia**")
            else:
                st.warning(
                    f"Models conflict — Model 1 says **{m1_label}**, Model 2 says **{m2_label}**. "
                    "Resolving by highest confidence..."
                )
                final_label, winning_model, confidence = resolve_conflict(
                    m1_label, m1_probs, m2_label, m2_probs
                )
                st.info(
                    f"{winning_model} wins with {confidence * 100:.1f}% confidence.  \n"
                    f"Final result: **{final_label} Pneumonia**"
                )


if __name__ == "__main__":
    main()
