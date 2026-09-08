import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(42)
n = 5000

X = pd.DataFrame({
    "amount": rng.uniform(5, 10000, n),
    "distance_from_home": rng.uniform(0, 1000, n),
    "distance_from_last_transaction": rng.uniform(0, 800, n),
    "ratio_to_median_purchase_price": rng.uniform(0.1, 10, n),
    "repeat_retailer": rng.integers(0, 2, n),
    "used_chip": rng.integers(0, 2, n),
    "used_pin_number": rng.integers(0, 2, n),
    "online_order": rng.integers(0, 2, n),
})

risk = (
    0.002 * X["amount"]
    + 0.01 * X["distance_from_home"]
    + 0.01 * X["distance_from_last_transaction"]
    + 0.5 * X["ratio_to_median_purchase_price"]
    + X["online_order"]
    + (1 - X["used_chip"])
    + (1 - X["used_pin_number"])
)

y = (risk > np.percentile(risk, 94)).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

reg = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    max_iter=500,
    random_state=42
)

reg.fit(X_train, y_train)

scores = reg.predict(X_test)
auc = roc_auc_score(y_test, scores)

print(f"test_auc={auc:.4f}")

joblib.dump(reg, "fraud-classifier.joblib")

print("Saved fraud-classifier.joblib")
