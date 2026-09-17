import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


MODEL_PATH = Path("fraud_model.onnx")

EXPECTED_FEATURE_COUNT = 8

FEATURE_NAMES = [
    "amount",
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]


def fail(message):
    print(f"\n❌ VALIDATION FAILED: {message}")
    sys.exit(1)


print("=" * 60)
print("Fraud Detection ONNX Model Validation")
print("=" * 60)


# ---------------------------------------------------------
# 1. Check that the ONNX model exists
# ---------------------------------------------------------

print("\n[1/5] Checking model file...")

if not MODEL_PATH.exists():
    fail(f"Model file not found: {MODEL_PATH}")

print(f"✅ Model found: {MODEL_PATH}")


# ---------------------------------------------------------
# 2. Validate ONNX structure
# ---------------------------------------------------------

print("\n[2/5] Validating ONNX structure...")

try:
    model = onnx.load(str(MODEL_PATH))
    onnx.checker.check_model(model)

except Exception as exc:
    fail(f"Invalid ONNX model: {exc}")

print("✅ ONNX structure is valid")


# ---------------------------------------------------------
# 3. Load model with ONNX Runtime
# ---------------------------------------------------------

print("\n[3/5] Loading model with ONNX Runtime...")

try:
    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

except Exception as exc:
    fail(f"ONNX Runtime could not load model: {exc}")

inputs = session.get_inputs()

if len(inputs) != 1:
    fail(
        f"Expected exactly 1 model input tensor, "
        f"but found {len(inputs)}"
    )

model_input = inputs[0]

print(f"Input name : {model_input.name}")
print(f"Input shape: {model_input.shape}")
print(f"Input type : {model_input.type}")


# ---------------------------------------------------------
# 4. Verify that model expects 8 features
# ---------------------------------------------------------

print("\n[4/5] Checking feature count...")

shape = model_input.shape

if len(shape) != 2:
    fail(
        f"Expected a 2-dimensional input such as (-1, 8), "
        f"but model reports {shape}"
    )

feature_dimension = shape[1]

if isinstance(feature_dimension, int):
    if feature_dimension != EXPECTED_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but model expects {feature_dimension}"
        )

print(f"✅ Model accepts {EXPECTED_FEATURE_COUNT} features")

print("\nExpected feature order:")

for number, feature in enumerate(FEATURE_NAMES, start=1):
    print(f"  {number}. {feature}")


# ---------------------------------------------------------
# 5. Run inference using RAW / HUMAN-READABLE values
# ---------------------------------------------------------

print("\n[5/5] Running inference tests...")

# Example 1:
# Lower-risk looking transaction.
#
# IMPORTANT:
# These are RAW feature values.
# They are NOT normalized before being sent to the model.

normal_transaction = np.array(
    [[
        45.00,    # amount
        5.0,      # distance_from_home
        3.0,      # distance_from_last_transaction
        0.80,     # ratio_to_median_purchase_price
        1.0,      # repeat_retailer
        1.0,      # used_chip
        1.0,      # used_pin_number
        0.0,      # online_order
    ]],
    dtype=np.float32,
)


# Example 2:
# Higher-risk looking transaction.
#
# Again, these are RAW human-readable values.

suspicious_transaction = np.array(
    [[
        950.00,   # amount
        120.0,    # distance_from_home
        80.0,     # distance_from_last_transaction
        6.50,     # ratio_to_median_purchase_price
        0.0,      # repeat_retailer
        0.0,      # used_chip
        0.0,      # used_pin_number
        1.0,      # online_order
    ]],
    dtype=np.float32,
)


def predict(name, transaction):

    print(f"\n{name}")
    print("-" * 45)

    print("Input:")
    for feature, value in zip(FEATURE_NAMES, transaction[0]):
        print(f"  {feature:<38} {value}")

    try:
        result = session.run(
            None,
            {model_input.name: transaction},
        )

    except Exception as exc:
        fail(f"Inference failed for {name}: {exc}")

    if not result:
        fail(f"Model returned no output for {name}")

    prediction = np.asarray(result[0])

    if prediction.size == 0:
        fail(f"Model returned an empty prediction for {name}")

    if not np.all(np.isfinite(prediction)):
        fail(
            f"Model returned NaN or infinite value for {name}: "
            f"{prediction}"
        )

    score = float(prediction.reshape(-1)[0])

    print(f"\nRaw model output: {score:.6f}")

    return score


normal_score = predict(
    "NORMAL TRANSACTION",
    normal_transaction,
)

suspicious_score = predict(
    "SUSPICIOUS TRANSACTION",
    suspicious_transaction,
)


# ---------------------------------------------------------
# Validation summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(f"Normal transaction score     : {normal_score:.6f}")
print(f"Suspicious transaction score : {suspicious_score:.6f}")

print("\nChecks:")
print("  ✅ ONNX file exists")
print("  ✅ ONNX structure is valid")
print("  ✅ ONNX Runtime can load the model")
print("  ✅ Model accepts 8 raw features")
print("  ✅ Normal transaction inference succeeded")
print("  ✅ Suspicious transaction inference succeeded")
print("  ✅ Predictions contain finite numeric values")

print("\n🎉 MODEL VALIDATION PASSED")
print("=" * 60)
