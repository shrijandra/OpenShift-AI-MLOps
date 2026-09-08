import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

reg = joblib.load("loan-default-risk.joblib")
initial_type = [("input", FloatTensorType([None, 5]))]
onx = convert_sklearn(reg, initial_types=initial_type)

with open("loan-model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
print("Converted to loan-model.onnx")
