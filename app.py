
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("svm_iris_model.pkl")
encoder = joblib.load("iris_encoder.pkl")

# Khởi tạo FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)

# Dữ liệu đầu vào
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Trang chính
@app.get("/")
def home():
    return {
        "message": "Iris Classification API is running!"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# API dự đoán
species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    prediction = int(model.predict(features)[0])

    return {
        "class_id": prediction,
        "prediction": species[prediction],
    }

