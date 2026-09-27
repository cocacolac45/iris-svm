import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ĐỌC DỮ LIỆU
df = pd.read_csv("Iris.csv")

FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = df[FEATURES]
y = df["Species"]

# MÃ HÓA NHÃN

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("Các lớp:", encoder.classes_)

# CHIA TRAIN / TEST

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

# CHUẨN HÓA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KHAI BÁO CÁC MÔ HÌNH VÀ LƯỚI SIÊU THAM SỐ

models = {
    "SVM": (
        SVC(probability=True, random_state=42),
        [
            # linear không dùng gamma nên tách riêng để khỏi chạy thừa
            {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
            {
                "kernel": ["rbf", "poly"],
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.1, 1],
            },
        ],
    ),  
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "newton-cg"],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
            "weights": ["uniform", "distance"],
            # minkowski (p=2) trùng euclidean nên bỏ; thay bằng manhattan
            "metric": ["euclidean", "manhattan"],
        },
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
    ),
    "LDA": (
        LinearDiscriminantAnalysis(),
        [
            {"solver": ["svd"]},
            {"solver": ["lsqr", "eigen"], "shrinkage": [None, "auto"]},
        ],
    ),
}

# HUẤN LUYỆN, ĐÁNH GIÁ, LƯU TỪNG MÔ HÌNH
for name, (estimator, param_grid) in models.items():
    print("\n" + "=" * 60)
    print(name.upper())
    print("=" * 60)

    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print("Siêu tham số tốt nhất:", grid.best_params_)
    print(f"Accuracy CV tốt nhất: {grid.best_score_:.4f}")
    print(f"Accuracy Test:        {accuracy:.4f}")

    print("\nMa trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred))

    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    filename = name.lower().replace(" ", "_") + "_iris_model.pkl"
    joblib.dump(best_model, filename)
    print(f"Đã lưu: {filename}")

# LƯU PREPROCESSING
joblib.dump(scaler, "iris_scaler.pkl")
joblib.dump(encoder, "iris_encoder.pkl")

print("\nĐã lưu: iris_scaler.pkl, iris_encoder.pkl")
