from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Header, Depends

import bcrypt
import joblib
import time
import secrets
import os
import psycopg2


# DATABASE SQL SERVER
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# SESSION
sessions = {}

# LOAD MODEL
scaler = joblib.load("iris_scaler.pkl")
encoder = joblib.load("iris_encoder.pkl")

# LOAD CÁC MÔ HÌNH HUẤN LUYỆN
models = {
    "SVM": joblib.load("svm_iris_model.pkl"),
    "Logistic Regression":
        joblib.load("logistic_regression_iris_model.pkl"),
    "KNN":
        joblib.load("knn_iris_model.pkl"),
    "Naive Bayes":
        joblib.load("naive_bayes_iris_model.pkl"),
    "LDA":
        joblib.load("lda_iris_model.pkl"),
}

# FASTAPI
app = FastAPI(
    title="Iris Classification API",
    description="Iris Classification with SQL Server",
    version="2.0.0",
)

security = HTTPBearer()

# STATIC PHOTO
app.mount(
    "/photo",
    StaticFiles(directory="photo"),
    name="photo"
)

# INPUT DATA
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    model_name: str = "SVM"

class RegisterInput(BaseModel):
    username: str
    password: str

class LoginInput(BaseModel):
    username: str
    password: str

class ChangePasswordInput(BaseModel):
    current_password: str
    new_password: str

# AUTHENTICATION HELPER
def get_current_user(
    credentials: HTTPAuthorizationCredentials
):
    token = credentials.credentials

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Token không hợp lệ"
        )

    if token not in sessions:
        raise HTTPException(
            status_code=401,
            detail="Phiên đăng nhập đã hết hạn hoặc không hợp lệ"
        )

    return sessions[token]

# TRANG CHỦ
@app.get("/", response_class=HTMLResponse)
def home():

    with open(
        "predict.html",
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()

# HEALTH CHECK
@app.get("/health")
def health():
    # Kiểm tra luôn database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1"
        )
        cursor.fetchone()
        cursor.close()
        conn.close()
        database_status = "connected"
    except Exception as e:
        database_status = f"error: {str(e)}"
    return {
        "status": "healthy",
        "database": database_status
    }

# REGISTER
@app.post("/register")
def register(data: RegisterInput):
    username = data.username.strip()
    password = data.password
    # Kiểm tra dữ liệu
    if not username:
        raise HTTPException(
            status_code=400,
            detail="Tên tài khoản không được để trống"
        )
    if not password:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu không được để trống"
        )
    if len(username) < 3:
        raise HTTPException(
            status_code=400,
            detail="Tên tài khoản phải có ít nhất 3 ký tự"
        )
    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu phải có ít nhất 6 ký tự"
        )
    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu không được vượt quá 72 bytes"
        )
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Kiểm tra username đã tồn tại
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE username = %s
            """,
            (username,)
        )
        existing_user = cursor.fetchone()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Tên tài khoản đã tồn tại"
            )
        # Hash password
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        # Insert user
        cursor.execute(
            """
            INSERT INTO users (
                username,
                password_hash
            )
            VALUES (%s, %s)
            """,
            (
                username,
                password_hash
            )
        )
        conn.commit()
        return {
            "message": "Đăng ký thành công",
            "username": username
        }

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi đăng ký: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()


# ====== LOGIN =========
@app.post("/login")
def login(data: LoginInput):

    username = data.username.strip()
    password = data.password

    if not username or not password:

        raise HTTPException(
            status_code=400,
            detail="Vui lòng nhập tài khoản và mật khẩu"
        )

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT
                id,
                username,
                password_hash
            FROM users
            WHERE username = %s
            """,
            (username,)
        )

        user = cursor.fetchone()

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Sai tài khoản hoặc mật khẩu"
            )

        user_id = user[0]
        db_username = user[1]
        password_hash = user[2]

        # Kiểm tra password
        if not bcrypt.checkpw(
                password.encode("utf-8"),
                password_hash.encode("utf-8")
        ):
            raise HTTPException(
                status_code=401,
                detail="Sai tài khoản hoặc mật khẩu"
            )

        # Tạo token
        token = secrets.token_urlsafe(32)

        sessions[token] = {
            "user_id": int(user_id),
            "username": db_username
        }

        return {
            "message": "Đăng nhập thành công",
            "token": token,
            "user_id": int(user_id),
            "username": db_username
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi đăng nhập: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()


# ======== LOGOUT =========
@app.post("/logout")
def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    token = credentials.credentials
    sessions.pop(token, None)

    return {"message": "Đăng xuất thành công"}

# ======== CURRENT USER ==========
@app.get("/me")
def get_me(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    user = get_current_user(credentials)

    return {
        "logged_in": True,
        "user_id": user["user_id"],
        "username": user["username"]
    }

# ============== CHANGE PASSWORD =============
@app.put("/change-password")
def change_password(
    data: ChangePasswordInput,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    user = get_current_user(credentials)
    user_id = user["user_id"]
    current_password = data.current_password
    new_password = data.new_password

    # Kiểm tra dữ liệu
    if not current_password:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng nhập mật khẩu hiện tại"
        )

    if not new_password:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng nhập mật khẩu mới"
        )

    if len(new_password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu mới phải có ít nhất 6 ký tự"
        )

    if len(new_password.encode("utf-8")) > 60:
        raise HTTPException(
            status_code=400,
            detail="Mật khẩu mới không được vượt quá 60 bytes"
        )

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Lấy mật khẩu hiện tại
        cursor.execute(
            """
            SELECT password_hash
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )

        user_data = cursor.fetchone()

        if not user_data:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy tài khoản"
            )

        password_hash = user_data[0]

        # Kiểm tra mật khẩu hiện tại
        if not bcrypt.checkpw(
            current_password.encode("utf-8"),
            password_hash.encode("utf-8")
        ):
            raise HTTPException(
                status_code=400,
                detail="Mật khẩu hiện tại không chính xác"
            )

        # Không cho đổi thành chính mật khẩu cũ
        if bcrypt.checkpw(
            new_password.encode("utf-8"),
            password_hash.encode("utf-8")
        ):
            raise HTTPException(
                status_code=400,
                detail="Mật khẩu mới phải khác mật khẩu hiện tại"
            )

        # Hash mật khẩu mới
        new_password_hash = bcrypt.hashpw(
            new_password.encode("utf-8"),
            bcrypt.gensalt()
        ).decode("utf-8")

        # Cập nhật database
        cursor.execute(
            """
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
            """,
            (
                new_password_hash,
                user_id
            )
        )

        conn.commit()
        return {"message": "Đổi mật khẩu thành công"}

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi đổi mật khẩu: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()

# =========== DELETE ACCOUNT =============
@app.delete("/account")
def delete_account(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    user = get_current_user(credentials)
    user_id = user["user_id"]
    token = credentials.credentials
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Xóa lịch sử dự đoán trước
        cursor.execute(
            """
            DELETE FROM prediction_history
            WHERE user_id = %s
            """,
            (user_id,)
        )

        # Xóa tài khoản
        cursor.execute(
            """
            DELETE FROM users
            WHERE id = %s
            """,
            (user_id,)
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy tài khoản"
            )
        conn.commit()

        # Xóa session hiện tại
        sessions.pop(token, None)

        return {"message": "Tài khoản đã được xóa thành công"}

    except HTTPException:
        conn.rollback()
        raise

    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xóa tài khoản: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()

# ======= PREDICT API ===========
@app.post("/predict")
def predict(
    data: IrisInput,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    user = get_current_user(credentials)
    user_id = user["user_id"]

    # ========== KIỂM TRA MODEL ===========
    if data.model_name not in models:
        raise HTTPException(
            status_code=400,
            detail="Mô hình không hợp lệ"
        )
    selected_model = models[data.model_name]

    # ============= TẠO DỮ LIỆU ĐẦU VÀO ===============
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]]

    # ============= SCALE DỮ LIỆU ============
    features_scaled = scaler.transform(features)

    # =========== ĐO THỜI GIAN CHẠY MODEL ===========
    start_time = time.perf_counter()

    # ========== DỰ ĐOÁN ==========
    prediction = int(selected_model.predict(features_scaled)[0])
    end_time = time.perf_counter()
    execution_time = (end_time - start_time)

    # =========== ĐỔI CLASS ID → TÊN LOÀI ============
    predicted_class = (encoder.inverse_transform([prediction])[0])

    # ============ TÍNH ĐỘ TIN CẬY =============
    confidence = None

    if hasattr(
        selected_model,
        "predict_proba"
    ):
        probabilities = (selected_model.predict_proba(features_scaled))
        confidence = float(probabilities.max())

    # ========= LƯU VÀO DATABASE ===========
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO prediction_history (
                user_id,
                model_name,
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
                prediction,
                confidence,
                execution_time
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                data.model_name,
                data.sepal_length,
                data.sepal_width,
                data.petal_length,
                data.petal_width,
                predicted_class,
                confidence,
                execution_time
            )
        )
        conn.commit()

    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Dự đoán thành công nhưng không thể lưu lịch sử: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()


    # ======== TRẢ KẾT QUẢ ========
    return {
        "class_id": prediction,
        "prediction": predicted_class,
        "model_name": data.model_name,
        "execution_time": execution_time,
        "confidence": confidence,
        "user_id": user_id
    }

# ========= GET HISTORY ============
@app.get("/history")
def get_history(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    user = get_current_user(credentials)
    user_id = user["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT
                id,
                model_name,
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
                prediction,
                confidence,
                execution_time,
                created_at
            FROM prediction_history
            WHERE user_id = %s
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,)
        )

        rows = cursor.fetchall()
        history = []

        for row in rows:
            history.append({
                "id": int(row[0]),
                "model": row[1],
                "sl": float(row[2]),
                "sw": float(row[3]),
                "pl": (row[4]),
                "pw": float(row[5]),
                "prediction": row[6],
                "confidence":
                    float(row[7])
                    if row[7] is not None
                    else None,
                "execution_time":
                    float(row[8])
                    if row[8] is not None
                    else None,
                "created_at":
                    row[9].isoformat()
                    if row[9] is not None
                    else None
            })

        return {
            "user_id": user_id,
            "history": history
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi lấy lịch sử: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()


# ========== DELETE ALL HISTORY ==============
@app.delete("/history")
def clear_history(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    user = get_current_user(credentials)
    user_id = user["user_id"]
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            DELETE FROM prediction_history
            WHERE user_id = %s
            """,
            (user_id,)
        )

        deleted_count = cursor.rowcount
        conn.commit()

        return {
            "message":
                "Đã xóa lịch sử",
            "deleted_count":
                deleted_count
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xóa lịch sử: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()

# ========= CHECK PASSWORD =============
class CheckPasswordInput(BaseModel):
    password: str

@app.post("/check-password")
def check_password(
    data: CheckPasswordInput,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):

    user = get_current_user(credentials)
    user_id = user["user_id"]
    password = data.password

    # Kiểm tra mật khẩu có được nhập hay không
    if not password:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng nhập mật khẩu"
        )

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Lấy mật khẩu đã hash trong database
        cursor.execute(
            """
            SELECT password_hash
            FROM users
            WHERE id = %s
            """,
            (user_id,)
        )

        user_data = cursor.fetchone()

        if not user_data:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy tài khoản"
            )

        password_hash = user_data[0]
        # Kiểm tra mật khẩu nhập vào
        password_correct = bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )

        return {"correct": password_correct}

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi kiểm tra mật khẩu: {str(e)}"
        )

    finally:
        cursor.close()
        conn.close()
