from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import time

# LOAD MODEL
model = joblib.load("svm_iris_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
encoder = joblib.load("iris_encoder.pkl")

# LOAD CÁC MÔ HÌNH HUẤN LUYỆN
models = {
    "SVM": joblib.load("svm_iris_model.pkl"),
    "Logistic Regression": joblib.load("logistic_regression_iris_model.pkl"),
    "KNN": joblib.load("knn_iris_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_iris_model.pkl"),
    "LDA": joblib.load("lda_iris_model.pkl"),
}

# FASTAPI
app = FastAPI(
    title="Iris Classification API",
    description="SVM model for the Iris dataset",
    version="1.0.0",
)

# Cho phép website truy cập ảnh trong folder photo
app.mount("/photo", StaticFiles(directory="photo"), name="photo")

# INPUT DATA
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    model_name: str = "SVM"

# TRANG CHỦ
@app.get("/", response_class=HTMLResponse)
def home():

    with open("predict.html", "r", encoding="utf-8") as f:
        return f.read()

# HEALTH CHECK
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# PREDICT API
@app.post("/predict")
def predict(data: IrisInput):

    # CHỌN MÔ HÌNH
    if data.model_name not in models:
        return {"error": "Mô hình không hợp lệ"}

    selected_model = models[data.model_name]

    # TẠO DỮ LIỆU ĐẦU VÀO
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    # SCALE DỮ LIỆU
    features_scaled = scaler.transform(features)

    # ĐO THỜI GIAN CHẠY MÔ HÌNH
    start_time = time.perf_counter()

    # DỰ ĐOÁN
    prediction = int(selected_model.predict(features_scaled)[0])
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # ĐỔI CLASS ID → TÊN LOÀI
    predicted_class = encoder.inverse_transform([prediction])[0]

    # TÍNH ĐỘ TIN CẬY
    confidence = None

    if hasattr(selected_model, "predict_proba"):
        probabilities = selected_model.predict_proba(features_scaled)
        confidence = float(probabilities.max())

    return {
        "class_id": prediction,
        "prediction": predicted_class,
        "model_name": data.model_name,
        "execution_time": execution_time,
        "confidence": confidence,
    }
