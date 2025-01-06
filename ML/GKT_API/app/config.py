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
