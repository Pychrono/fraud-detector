# 🔐 Fraud Detection Dashboard

This is a real-time fraud detection web application that uses machine learning to predict the likelihood of fraudulent transactions, featuring an interactive dashboard powered by Flask and Chart.js.

---

## 🚀 Live Demo  
👉 [fraud-detector-vuz6.onrender.com](https://fraud-detector-vuz6.onrender.com)

---

## ✨ Features

✅ Quick Check — use simplified inputs for fast predictions  
✅ Advanced Mode — full control with 28 fraud-related variables  
✅ Real-time animated charts:
- 🧁 Doughnut chart for fraud distribution
- 📊 Bar chart for transaction amounts
- 📈 Line chart showing simulated fraud risk

✅ Responsive dark-themed UI (mobile-friendly)  
✅ Trained on real 2023 Kaggle dataset

---

## ⚙️ Tech Stack

- **Backend:** Python, Flask, scikit-learn, pandas  
- **Frontend:** HTML, CSS (custom), Chart.js  
- **Deployment:** Render, GitHub

---

## 🧠 Machine Learning

Two models were trained:

- 🎯 **Full Model** — uses PCA-transformed features (V1–V28) from real credit card fraud data  
- ⚡ **Quick Check Model** — trained on `Amount`, `is_foreign`, `is_high_risk_country`, `used_chip`

Both models were saved with `joblib` and loaded into the Flask API.

---

## 📦 Project Structure

