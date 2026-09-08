import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score
import joblib

rng = np.random.default_rng(7)
n = 8000

credit_score = np.clip(rng.normal(650, 80, n), 300, 850)
debt_to_income = np.clip(rng.normal(0.35, 0.15, n), 0, 1)
loan_amount = np.clip(rng.normal(15000, 8000, n), 1000, 50000)
income = np.clip(rng.normal(65000, 25000, n), 15000, 200000)
employment_length = np.clip(rng.exponential(6, n), 0, 40)

# Plausible relationship: lower score, higher debt ratio, bigger loan,
# lower income, shorter employment -> higher default risk
logit = (
    -0.03 * (credit_score - 650)
    + 4.0 * (debt_to_income - 0.35)
    + 0.00002 * (loan_amount - 15000)
    - 0.00001 * (income - 65000)
    - 0.05 * (employment_length - 5)
    + rng.normal(0, 0.5, n)
)
prob_default = 1 / (1 + np.exp(-logit))
label = rng.binomial(1, prob_default)

# Fixed normalization constants -- used identically at inference time
X = np.column_stack([
    (credit_score - 300) / 550,
    debt_to_income,
    loan_amount / 50000,
    income / 200000,
    employment_length / 40,
])

X_train, X_test, y_train, y_test = train_test_split(
    X, label, test_size=0.2, stratify=label, random_state=7
)

reg = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=7)
reg.fit(X_train, y_train.astype(float))

auc = roc_auc_score(y_test, reg.predict(X_test))
print(f"test_auc={auc:.4f}")

joblib.dump(reg, "loan-default-risk.joblib")
print("Saved model to loan-default-risk.joblib")
