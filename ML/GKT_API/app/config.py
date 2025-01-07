import os

# BASE_DIR 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # MLflow Tracking URI
# MLFLOW_TRACKING_URI = os.getenv(
#     "MLFLOW_SERVER_URI", "http://bigdata9:bigdata9-@mane.my/mlflow/"
# )

# 모델 경로
MODEL_PATH = "/code/app/model/model.pth"

# 데이터 경로
LOCAL_CSV_PATH = "/code/app/filtered_combined_user_data.csv"
LOCAL_PARQUET_PATH = "/code/app/filtered_combined_user_data.parquet"

# S3 정보
S3_BUCKET = "big9-project-01"
S3_FILE_KEY = "data/tbl_app_testhisdtl/filtered_combined_user_data.parquet"

# AWS RDS 연결 정보 설정
DB_CONFIG = {
    "host": "bigdata-team-01.cfsgom2iusui.ap-northeast-2.rds.amazonaws.com",  # RDS 엔드포인트
    "port": 5432,                               # PostgreSQL 기본 포트
    "database": "math_db",                # 데이터베이스 이름
    "user": "postgres",                    # 사용자 이름
    "password": "bigdata9-",                # 비밀번호
}