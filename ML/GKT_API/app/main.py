from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import mlflow.pytorch
import os
import polars as pl
import sys
import boto3
import time
import numpy as np
import concurrent.futures

# BASE_DIR 설정 및 sys.path에 모델 경로 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "model"))

print("Current sys.path:", sys.path)

# MLflow Tracking URI 설정
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_SERVER_URI", "http://bigdata9:bigdata9-@mane.my/mlflow/")
)


def get_csv_file(local_path, s3_bucket_name, s3_key):
    """
    CSV 파일을 로컬에서 읽거나 S3에서 다운로드 후 읽는 함수.

    Args:
        local_path (str): 로컬 파일 경로.
        s3_bucket_name (str): S3 버킷 이름.
        s3_key (str): S3에서 파일의 경로.

    Returns:
        pl.DataFrame: Polars DataFrame.
    """
    if os.path.exists(local_path):
        print(f"로컬에서 파일을 읽습니다: {local_path}")
        return pl.read_csv(local_path)

    print(f"로컬 파일이 없습니다. S3에서 파일을 다운로드합니다: {s3_bucket_name}/{s3_key}")
    try:
        # S3 클라이언트 생성
        s3 = boto3.client("s3")

        # 파일 다운로드
        s3.download_file(s3_bucket_name, s3_key, local_path)
        print(f"파일이 성공적으로 다운로드되었습니다: {local_path}")

        # CSV 파일 읽기
        return pl.read_csv(local_path)

    except Exception as e:
        raise RuntimeError(f"S3에서 파일을 다운로드하는 중 에러가 발생했습니다: {e}")


def get_parquet_file(local_path, s3_bucket_name, s3_key):
    """
    Parquet 파일을 로컬에서 읽거나 S3에서 다운로드 후 읽는 함수.

    Args:
        local_path (str): 로컬 파일 경로.
        s3_bucket_name (str): S3 버킷 이름.
        s3_key (str): S3에서 파일의 경로.

    Returns:
        pl.DataFrame: Polars DataFrame.
    """
    if os.path.exists(local_path):
        print(f"로컬에서 파일을 읽습니다: {local_path}")
        return pl.read_parquet(local_path)

    print(f"로컬 파일이 없습니다. S3에서 파일을 다운로드합니다: {s3_bucket_name}/{s3_key}")
    try:
        # S3 클라이언트 생성
        s3 = boto3.client("s3")

        # S3에서 파일 다운로드
        s3.download_file(s3_bucket_name, s3_key, local_path)
        print(f"파일이 성공적으로 다운로드되었습니다: {local_path}")

        # Parquet 파일 읽기
        return pl.read_parquet(local_path)

    except Exception as e:
        raise RuntimeError(f"S3에서 파일을 다운로드하는 중 에러가 발생했습니다: {e}")


# 예제 사용
local_csv_path = "/code/app/filtered_combined_user_data.csv"
local_parquet_path = "/code/app/filtered_combined_user_data.parquet"
s3_bucket = "big9-project-01"
s3_file_key = "data/tbl_app_testhisdtl/filtered_combined_user_data.parquet"

try:
    # data = get_csv_file(local_csv_path, s3_bucket, s3_file_key)
    data = get_parquet_file(local_parquet_path, s3_bucket, s3_file_key)
    print("데이터가 성공적으로 로드되었습니다.")
except RuntimeError as e:
    print(f"에러: {e}")


def load_model():
    """
    MLflow에서 모델을 로드하는 함수.

    Returns:
        torch.nn.Module: 로드된 PyTorch 모델.
    """
    logged_model = "runs:/446b1a8e75ff4263a59f168a5605ba90/best_model"
    model = mlflow.pytorch.load_model(logged_model, map_location=torch.device("cpu"))
    model.eval()
    return model


model = load_model()

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    docs_url="/docs",  # Swagger UI 경로
    openapi_url="/api/gkt/openapi.json",  # OpenAPI 스펙 경로
)


@app.get("/health")
def health_check():
    """
    Health Check 엔드포인트.
    서버가 정상적으로 작동하는지 확인하기 위해 사용.
    """
    return {"status": "ok"}


# 입력 데이터 모델 정의
class InputData(BaseModel):
    user_id: str
    skill_list: list
    correct_list: list


@app.post("/api/gkt")
async def predict(input_data: InputData):
    """
    예측 API 엔드포인트.
    입력 데이터를 받아 모델 예측 결과를 반환.

    Args:
        input_data (InputData): 예측에 필요한 입력 데이터.

    Returns:
        dict: 예측 결과.
    """
    try:
        start_time = time.time()  # 전체 프로세스 시작 시간

        # Step 1 - 유저 데이터 필터링
        user_id = input_data.user_id
        step1_start = time.time()
        user_data = data.filter(pl.col("UserID") == user_id)
        step1_end = time.time()
        print(f"Step 1 (유저 데이터 필터링): {step1_end - step1_start:.4f} 초")

        # Step 2 - 데이터 정렬
        step2_start = time.time()
        user_data = user_data.sort("CreDate")  # "CreDate" 기준 정렬
        step2_end = time.time()
        print(f"Step 2 (데이터 정렬): {step2_end - step2_start:.4f} 초")

        # Step 3 - Enumerate skill id
        step3_start = time.time()
        skill_map = user_data["f_mchapter_id"].unique().sort().to_list()
        skill_map_dict = {value: idx for idx, value in enumerate(skill_map)}
        user_data = user_data.with_columns(
            pl.col("f_mchapter_id")
            .replace(skill_map_dict)
            .cast(pl.Int32)
            .alias("skill")
        )
        step3_end = time.time()
        print(f"Step 3 (스킬 매핑): {step3_end - step3_start:.4f} 초")

        # Step 4 - Correct 변환
        step4_start = time.time()
        user_data = user_data.with_columns(
            pl.col("Correct").replace({"O": 1, "X": 0}).cast(pl.Int32).alias("Correct")
        )
        step4_end = time.time()
        print(f"Step 4 (Correct 변환): {step4_end - step4_start:.4f} 초")

        # Step 5 - Synthetic Feature 생성
        step5_start = time.time()
        user_data = user_data.with_columns(
            (pl.col("skill") * 2 + pl.col("Correct")).alias("skill_with_answer")
        )
        step5_end = time.time()
        print(f"Step 5 (Synthetic Feature 생성): {step5_end - step5_start:.4f} 초")

        # Step 6 - 입력 스킬 매핑
        step6_start = time.time()
        next_skills = [skill_map_dict.get(skill, -1) for skill in input_data.skill_list]
        if -1 in next_skills:
            raise HTTPException(
                status_code=400,
                detail="One or more skills in skill_list are not present in the data.",
            )
        step6_end = time.time()
        print(f"Step 6 (입력 스킬 매핑): {step6_end - step6_start:.4f} 초")

        # Step 7 - 모델 예측
        # CPU 스레드 수 설정
        cpu_core_count = os.cpu_count()  # CPU 코어 수 가져오기
        max_threads = max(1, cpu_core_count // 2)  # 코어 수의 절반만 사용
        torch.set_num_threads(max_threads)
        print(f"PyTorch에서 사용하는 스레드 수: {max_threads}개")

        def prepare_data(user_data, next_skills, input_data):
            """
            입력 데이터를 준비하는 함수.

            Args:
                user_data (pl.DataFrame): 유저 데이터.
                next_skills (list): 다음 스킬 리스트.
                input_data (InputData): 입력 데이터.

            Returns:
                tuple: features_tensor, questions_tensor
            """
            try:
                features = np.array(
                    user_data["skill_with_answer"].to_list(), dtype=np.int64
                )
                questions = np.array(user_data["skill"].to_list(), dtype=np.int64)

                next_skills_array = np.array(next_skills, dtype=np.int64)
                correct_list_array = np.array(input_data.correct_list, dtype=np.int64)

                # features와 questions 확장
                features = np.concatenate(
                    [features, next_skills_array * 2 + correct_list_array]
                )
                questions = np.concatenate([questions, next_skills_array])

                # 텐서 생성 (numpy → torch 텐서)
                features_tensor = torch.from_numpy(features).unsqueeze(0)
                questions_tensor = torch.from_numpy(questions).unsqueeze(0)

                return features_tensor, questions_tensor
            except Exception as e:
                print(f"입력 데이터 준비 중 오류 발생: {e}")
                raise

        def model_prediction(
            active_model, features_tensor, questions_tensor, next_skills
        ):
            """
            모델 예측을 수행하는 함수.

            Args:
                active_model (torch.nn.Module): 예측에 사용할 모델.
                features_tensor (torch.Tensor): 입력 피처 텐서.
                questions_tensor (torch.Tensor): 입력 질문 텐서.
                next_skills (list): 다음 스킬 리스트.

            Returns:
                torch.Tensor: 예측 결과.
            """
            try:
                with torch.no_grad():
                    pred_res, _, _, _ = active_model(features_tensor, questions_tensor)
                    next_preds = pred_res.squeeze(0)[-len(next_skills) :]
                return next_preds
            except Exception as e:
                print(f"모델 예측 중 오류 발생: {e}")
                raise

        # Step 7 - 모델 예측
        step7_start = time.time()

        # 입력 데이터 확인 및 변환
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_data = executor.submit(
                prepare_data, user_data, next_skills, input_data
            )
            features_tensor, questions_tensor = future_data.result()

        active_model = model  # TorchScript 변환 실패 시 기본 모델 사용

        # 모델 예측
        next_preds = model_prediction(
            active_model, features_tensor, questions_tensor, next_skills
        )

        step7_end = time.time()
        print(f"Step 7 (모델 예측): {step7_end - step7_start:.4f} 초")

        # 결과 생성
        step8_start = time.time()
        result = [
            {skill: next_preds[idx].item()}
            for idx, skill in enumerate(input_data.skill_list)
        ]
        step8_end = time.time()
        print(f"Step 8 (결과 생성): {step8_end - step8_start:.4f} 초")

        # 전체 소요 시간
        end_time = time.time()
        print(f"전체 소요 시간: {end_time - start_time:.4f} 초")

        return {"predictions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
