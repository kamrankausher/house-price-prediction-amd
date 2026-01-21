import streamlit as st
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data (for scaler stats)
df = pd.read_csv("../data/housing.csv")
df = df.drop(columns=["Address"])

X = df.drop("Price", axis=1)
y = df["Price"]

scaler = StandardScaler()
scaler.fit(X)

# Device (CPU now, AMD GPU later)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
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

# Load model
model = HousePriceModel(X.shape[1])
state_dict = torch.load("../models/house_price_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# UI
st.title("🏠 House Price Prediction (AMD ROCm Ready)")

inputs = []
for col in X.columns:
    val = st.number_input(col, float(X[col].min()), float(X[col].max()))
    inputs.append(val)

if st.button("Predict Price"):
    scaled = scaler.transform([inputs])
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(tensor)

    y_mean = y.mean()
    y_std = y.std()
    price = (pred.cpu().numpy()[0][0] * y_std) + y_mean

    st.success(f"Estimated House Price: ₹ {price:,.2f}")
