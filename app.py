import streamlit as st
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

# ===============================
# BACKGROUND + THEME (ONLINE IMAGE)
# ===============================
st.markdown("""
<style>
.stApp {
    background:
        linear-gradient(rgba(12,0,24,0.85), rgba(30,8,60,0.85)),
        url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c");
    background-size: cover;
    background-position: center;
}

/* 3D GOLD TITLE */
.title {
    font-size: 58px;
    font-weight: 900;
    color: #d4af37;
    text-align: center;
    letter-spacing: 1.5px;
    text-shadow:
        1px 1px 0 #b8962e,
        2px 2px 0 #9c7c1f,
        3px 3px 0 #7f6416;
    margin-bottom: 45px;
}

/* Card styling */
.card {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 26px;
    margin-bottom: 28px;
    animation: fadeIn 0.9s ease;
}

/* Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(18px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Buttons */
.stButton>button {
    background: #d4af37;
    color: black;
    font-weight: 700;
    border-radius: 12px;
    padding: 0.7rem 1.8rem;
    transition: 0.3s;
}

.stButton>button:hover {
    background: #e8c75c;
}

/* Labels */
label {
    color: #f0d878 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("data/housing.csv")
df = df.drop(columns=["Address"])

X = df.drop("Price", axis=1)
y = df["Price"]

scaler = StandardScaler()
scaler.fit(X)

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# MODEL
# ===============================
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

model = HousePriceModel(X.shape[1])
state_dict = torch.load("models/house_price_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.markdown("## 🧮 House Parameters")

inputs = []
for col in X.columns:
    val = st.sidebar.slider(
        col,
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    inputs.append(val)

# ===============================
# TITLE
# ===============================
st.markdown('<div class="title">🏠 House Price Prediction</div>', unsafe_allow_html=True)

# ===============================
# INPUT SUMMARY
# ===============================
st.markdown('<div class="card">📊 <b>Input Summary</b>', unsafe_allow_html=True)
st.dataframe(pd.DataFrame([inputs], columns=X.columns))
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# PREDICTION ENGINE
# ===============================
st.markdown('<div class="card">💰 <b>Prediction Engine</b>', unsafe_allow_html=True)

if st.button("🚀 Predict House Price"):
    scaled = scaler.transform([inputs])
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(tensor).cpu().numpy()[0][0]

    # Convert back to original scale
    price = (pred * y.std()) + y.mean()

    # Unit logic
    if price >= 1e7:
        display_price = f"₹ {price/1e7:.2f} Crores"
    else:
        display_price = f"₹ {price/1e5:.2f} Lakhs"

    st.markdown(f"### 🏷️ Estimated Property Value: **{display_price}**")
    st.success("Prediction completed successfully!")

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# IMAGE GALLERY
# ===============================
st.markdown('<div class="card">🏘️ <b>Property Inspirations</b>', unsafe_allow_html=True)

images = [
    "https://images.unsplash.com/photo-1570129477492-45c003edd2be",
    "https://images.unsplash.com/photo-1598928506311-c55ded91a20c",
    "https://images.unsplash.com/photo-1605276373954-0c4a0dac5b12",
    "https://images.unsplash.com/photo-1580587771525-78b9dba3b914"
]

cols = st.columns(4)
for col, img in zip(cols, random.sample(images, 4)):
    col.image(img, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# QUOTE SECTION
# ===============================
st.markdown("""
<div class="card" style="text-align:center;">
✨ <i>"Real estate cannot be lost or stolen — it is the foundation of long-term wealth."</i>
</div>
""", unsafe_allow_html=True)
