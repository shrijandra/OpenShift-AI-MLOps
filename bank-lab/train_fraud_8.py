import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score


# ============================================================
# 1. GENERATE SYNTHETIC FRAUD TRANSACTION DATA
# ============================================================

# Fixed random seed makes the demo reproducible.
rng = np.random.default_rng(42)

# Generate 5,000 synthetic credit-card transactions.
n = 5000

# Create 8 transaction features used by the fraud model.
X = pd.DataFrame({

    # Transaction amount in dollars.
    "amount": rng.uniform(5, 10000, n),

    # Distance of transaction from customer's home.
    "distance_from_home": rng.uniform(0, 1000, n),

    # Distance from the location of the previous transaction.
    "distance_from_last_transaction": rng.uniform(0, 800, n),

    # Current purchase compared with customer's typical purchase amount.
    "ratio_to_median_purchase_price": rng.uniform(0.1, 10, n),

    # Binary features: 1 = Yes, 0 = No.
    "repeat_retailer": rng.integers(0, 2, n),
    "used_chip": rng.integers(0, 2, n),
    "used_pin_number": rng.integers(0, 2, n),
    "online_order": rng.integers(0, 2, n),
})


# ============================================================
# 2. CREATE SYNTHETIC FRAUD-RISK BEHAVIOR
# ============================================================

# Since this is synthetic data, define rules that make certain
# transaction characteristics contribute to higher fraud risk.
#
# Examples:
#   Higher transaction amount             -> higher risk
#   Greater distance from home            -> higher risk
#   Unusually large purchase              -> higher risk
#   Online transaction                    -> higher risk
#   No chip / no PIN                      -> higher risk

risk = (
    0.002 * X["amount"]
    + 0.01 * X["distance_from_home"]
    + 0.01 * X["distance_from_last_transaction"]
    + 0.5 * X["ratio_to_median_purchase_price"]
    + X["online_order"]
    + (1 - X["used_chip"])
    + (1 - X["used_pin_number"])
)


# ============================================================
# 3. CREATE THE TARGET LABEL
# ============================================================

# Mark approximately the highest-risk 6% of transactions as fraud.
#
# y = 1 -> Fraud / higher-risk transaction
# y = 0 -> Normal transaction

y = (risk > np.percentile(risk, 94)).astype(float)


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TEST SETS
# ============================================================

# 80% of the data is used to train the model.
# 20% is held back for model evaluation.
#
# stratify=y preserves approximately the same fraud ratio
# in both the training and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ============================================================
# 5. TRAIN THE FRAUD-RISK MODEL
# ============================================================

# MLPRegressor is a feed-forward neural network.
#
# Architecture:
#   8 input features
#        ↓
#   32-neuron hidden layer
#        ↓
#   16-neuron hidden layer
#        ↓
#   Continuous fraud-risk score
#
# Higher output score = higher predicted fraud risk.

reg = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    max_iter=500,
    random_state=42
)

reg.fit(X_train, y_train)


# ============================================================
# 6. EVALUATE MODEL PERFORMANCE
# ============================================================

# Generate risk scores for transactions the model did not train on.
scores = reg.predict(X_test)

# ROC AUC measures how well the model ranks fraud transactions
# above normal transactions.
#
# 0.5 = approximately random ranking
# 1.0 = perfect separation
auc = roc_auc_score(y_test, scores)

print(f"test_auc={auc:.4f}")


# ============================================================
# 7. SAVE THE TRAINED MODEL
# ============================================================

# Save the trained Scikit-learn model as a Joblib artifact.
#
# Next MLOps step:
# JOBLIB -> ONNX -> MinIO/S3 -> OpenShift AI -> OpenVINO

joblib.dump(reg, "fraud-classifier.joblib")

print("Saved fraud-classifier.joblib")
