from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("svm_iris_model.pkl")
encoder = joblib.load("iris_encoder.pkl")

# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)

# Cho phép website truy cập ảnh trong folder photo
app.mount("/photo", StaticFiles(directory="photo"), name="photo")


# =========================
# INPUT DATA
# =========================

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# =========================
# CLASS
# =========================

species = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


# =========================
# TRANG CHỦ
# =========================

@app.get("/", response_class=HTMLResponse)
def home():

    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# =========================
# TRANG DỰ ĐOÁN
# =========================

@app.get("/predict-page", response_class=HTMLResponse)
def predict_page():

    with open("predict.html", "r", encoding="utf-8") as f:
        return f.read()


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================
# PREDICT API
# =========================

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
