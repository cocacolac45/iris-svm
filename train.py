import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Đọc dữ liệu
df = pd.read_csv("Iris.csv")

# Chọn 4 đặc trưng
X = df[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]

# Nhãn
y = df["Species"]

# Mã hóa nhãn
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("Các lớp:", encoder.classes_)

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Chuẩn hoá dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== TÌM SIÊU THAM SỐ TỐI ƯU BẰNG GRIDSEARCHCV =====
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=5,                # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,           # dùng đa luồng cho nhanh
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("\n===== KẾT QUẢ GRIDSEARCH =====")
print("Siêu tham số tốt nhất:", grid_search.best_params_)
print("Accuracy CV tốt nhất:", grid_search.best_score_)

# Lấy mô hình tốt nhất
model = grid_search.best_estimator_

# ===== ĐÁNH GIÁ TRÊN TẬP TEST =====
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n===== KẾT QUẢ TRÊN TẬP TEST =====")
print("Accuracy:", accuracy)
print("\nMa trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))
print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ===== CROSS-VALIDATION TRÊN TOÀN BỘ DỮ LIỆU (để kiểm tra độ ổn định) =====
X_scaled_full = scaler.fit_transform(X)
cv_scores = cross_val_score(model, X_scaled_full, y_encoded, cv=5)
print("\n===== CROSS-VALIDATION (5-fold) TRÊN TOÀN BỘ DỮ LIỆU =====")
print("Accuracy từng fold:", cv_scores)
print(f"Accuracy trung bình: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ===== LƯU MÔ HÌNH =====
joblib.dump(model, "svm_iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")   # nhớ lưu cả scaler!
joblib.dump(encoder, "iris_encoder.pkl")

print("\nĐã lưu model, scaler và encoder!")
