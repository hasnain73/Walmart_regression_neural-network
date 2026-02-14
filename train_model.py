"""
Walmart Sales Prediction - Neural Network Regression
Simple NN model for predicting weekly sales
Author: ML Engineer
Date: 2026-02-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("WALMART SALES PREDICTION - NEURAL NETWORK")
print("=" * 60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/7] Loading Dataset...")
# Note: Download dataset from Kaggle first
# Place Walmart_Sales.csv in the same directory
df = pd.read_csv('Walmart_Sales.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:\n{df.head()}")

# ============================================
# 2. DATA PREPROCESSING
# ============================================
print("\n[2/7] Preprocessing Data...")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Fill missing values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop unnecessary columns (if Store or Date are just IDs)
# Keep only relevant features
columns_to_drop = []
if 'Store' in df.columns:
    # We'll encode Store as a feature
    pass
if 'Date' in df.columns:
    columns_to_drop.append('Date')  # Drop date for simplicity

df = df.drop(columns=columns_to_drop, errors='ignore')

# Separate features and target
target_col = 'Weekly_Sales'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ============================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================
print("\n[3/7] Encoding Categorical Features...")

# Find categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ============================================
# 4. NORMALIZE FEATURES
# ============================================
print("\n[4/7] Normalizing Features...")

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 5. BUILD NEURAL NETWORK
# ============================================
print("\n[5/7] Building Neural Network...")

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu', name='hidden_layer_1'),
    layers.Dense(32, activation='relu', name='hidden_layer_2'),
    layers.Dense(1, activation='linear', name='output_layer')
], name='Walmart_Sales_NN')

# Compile model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nModel Summary:")
model.summary()

# ============================================
# 6. TRAIN MODEL
# ============================================
print("\n[6/7] Training Model (20 epochs)...")

history = model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ============================================
# 7. EVALUATE MODEL
# ============================================
print("\n[7/7] Evaluating Model...")

# Predictions
y_pred = model.predict(X_test_scaled).flatten()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION METRICS")
print("=" * 60)
print(f"Test MSE:  {mse:,.2f}")
print(f"Test MAE:  {mae:,.2f}")
print(f"Test RMSE: {rmse:,.2f}")
print(f"R² Score:  {r2:.4f}")
print("=" * 60)

# ============================================
# 8. PLOT TRAINING HISTORY
# ============================================
print("\nGenerating Training Plots...")

plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Predicted vs Actual
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.title('Predicted vs Actual Sales', fontsize=14, fontweight='bold')
plt.xlabel('Actual Weekly Sales')
plt.ylabel('Predicted Weekly Sales')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved training_results.png")
plt.show()

# ============================================
# 9. SAVE MODEL AND PREPROCESSORS
# ============================================
print("\nSaving Model and Preprocessors...")

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model
model.save('models/model.h5')
print("✓ Saved models/model.h5")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved models/scaler.pkl")

# Save label encoders if any
if label_encoders:
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    print("✓ Saved models/label_encoders.pkl")

# Save feature names for reference
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')
print("✓ Saved models/feature_names.pkl")

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print("\nNext Steps:")
print("1. Check training_results.png for visualizations")
print("2. Model files saved in ./models/ directory")
print("3. Run app.py to test Gradio interface")
print("4. Deploy to Hugging Face Spaces")
print("=" * 60)
