import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
df = pd.read_csv('../fraud-detector/data/creditcard_2023.csv')

# Rename 'Class' to 'is_fraud' just for consistency in your app
df.rename(columns={'Class': 'is_fraud'}, inplace=True)

# Define features and target
features = [col for col in df.columns if col not in ['id', 'is_fraud']]  # exclude 'id' and 'Class'
target = 'is_fraud'

X = df[features]
y = df[target]

# Optional: balance the dataset (fraud cases are rare!)
# df_fraud = df[df.is_fraud == 1]
# df_non_fraud = df[df.is_fraud == 0].sample(len(df_fraud))
# df_balanced = pd.concat([df_fraud, df_non_fraud])
# X = df_balanced[features]
# y = df_balanced[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, 'fraud_model.pkl')
print("✅ Model saved as fraud_model.pkl")
