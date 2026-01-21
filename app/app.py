import streamlit as st
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="House Price Prediction (AMD ROCm Ready)",
    page_icon="🏠",
    layout="centered"
)

# -------------------------------
# Load data (for scaler)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/housing.csv")
    df = df.drop(columns=["Address"])
    return df

df = load_data()

X = df.drop("Price", axis=1)

# -------------------------------
# Fit scaler on full dataset
# -------------------------------
scaler = StandardScaler()
scaler.fit(X)

# -------------------------------
# Device (CPU on Streamlit, AMD GPU ready locally)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model definition (MUST MATCH TRAINING)
# -------------------------------
class HousePriceModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    model = HousePriceModel(X.shape[1])
    state_dict = torch.load("models/house_price_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🏠 House Price Prediction")
st.caption("PyTorch model • AMD ROCm-ready inference design")

st.markdown("### Enter house details")

inputs = []
for col in X.columns:
    val = st.number_input(
        label=col,
        min_value=float(X[col].min()),
        max_value=float(X[col].max()),
        value=float(X[col].mean())
    )
    inputs.append(val)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    scaled = scaler.transform([inputs])
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(tensor)

    price = pred.cpu().numpy()[0][0]

    st.success(f"💰 Estimated House Price: ₹ {price:,.2f}")

st.markdown("---")
st.markdown(
    "**Tech Stack:** PyTorch • Streamlit • AMD ROCm-ready architecture"
)
