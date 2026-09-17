from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("svm_iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
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
# TRANG CHỦ
# =========================


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

    # =========================
    # TẠO DỮ LIỆU ĐẦU VÀO
    # =========================

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    # =========================
    # SCALE DỮ LIỆU
    # =========================

    features_scaled = scaler.transform(features)

    # =========================
    # DỰ ĐOÁN
    # =========================

    prediction = int(model.predict(features_scaled)[0])

    # =========================
    # ĐỔI CLASS ID → TÊN LOÀI
    # =========================

    predicted_class = encoder.inverse_transform([prediction])[0]

    return {
        "class_id": prediction,
        "prediction": predicted_class,
    }
