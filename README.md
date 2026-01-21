# 🏠 House Price Prediction — AMD ROCm Ready ML Project

## 📌 Project Overview
This project implements an **end-to-end Machine Learning pipeline** for predicting house prices using PyTorch.

The system is designed to be **device-agnostic** and **AMD ROCm compatible**, meaning the same inference code can run on:
- CPU (local development)
- AMD GPUs (via ROCm on AMD Developer Cloud)

---

## 🧠 Key Features
- End-to-end ML workflow (EDA → Training → Evaluation → Inference)
- PyTorch neural network model
- Proper feature scaling & inverse-scaling
- ROCm-ready inference logic
- Clean project structure
- Streamlit-ready deployment

---

## 🧱 Tech Stack
- Python 3.10
- PyTorch
- Scikit-learn
- Pandas, NumPy
- Jupyter Notebook
- Streamlit
- Git & GitHub

---

## 📂 Project Structure

house-price-prediction-amd/
│
├── data/
│ └── housing.csv
│
├── notebooks/
│ ├── 01_data_understanding.ipynb
│ ├── 07_rocm_ready_inference.ipynb
│
├── models/
│ └── house_price_model.pt
│
├── app/
│ └── app.py
│
├── requirements.txt
├── README.md
└── screenshots/


---

## 🚀 ROCm & AMD GPU Compatibility

The inference notebook uses the following device logic:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


This allows:

CPU execution during development

Seamless AMD GPU execution using ROCm without code changes

This reflects real-world, production-grade ML deployment practices.

📊 Results

Model trained using PyTorch

Predictions inverse-scaled to real house prices

Clear comparison between actual and predicted prices

🔮 Future Improvements

Run inference on AMD GPU (ROCm) cloud instance

Hyperparameter tuning

Model monitoring

CI/CD pipeline

👤 Author

Kamran Kausher
B.Tech CSE | Data Science & GenAI