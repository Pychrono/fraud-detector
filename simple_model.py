import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# === Load your Kaggle dataset ===
df = pd.read_csv("data/creditcard_2023.csv")  # Update path if it's elsewhere
df.rename(columns={'Class': 'is_fraud'}, inplace=True)

# === Add mock features for quick check ===
np.random.seed(42)
df['is_foreign'] = np.random.randint(0, 2, size=len(df))
df['is_high_risk_country'] = np.random.randint(0, 2, size=len(df))
df['used_chip'] = np.random.randint(0, 2, size=len(df))

# === Sample a small balanced set (optional but smart) ===
fraud = df[df['is_fraud'] == 1].sample(2500, random_state=42)
non_fraud = df[df['is_fraud'] == 0].sample(2500, random_state=42)
df_balanced = pd.concat([fraud, non_fraud])

# === Define features & labels ===
features = ['Amount', 'is_foreign', 'is_high_risk_country', 'used_chip']
X = df_balanced[features]
y = df_balanced['is_fraud']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Train model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Save model ===
joblib.dump(model, "models/simple_fraud_model.pkl")
print("✅ Simple model saved to models/simple_fraud_model.pkl")
