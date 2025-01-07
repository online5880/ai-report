import os
import boto3
import polars as pl
import psycopg2

def get_csv_file(local_path, s3_bucket_name, s3_key):
    if os.path.exists(local_path):
        print(f"로컬에서 파일을 읽습니다: {local_path}")
        return pl.read_csv(local_path)

    print(f"로컬 파일이 없습니다. S3에서 파일을 다운로드합니다: {s3_bucket_name}/{s3_key}")
    try:
        s3 = boto3.client("s3")
        s3.download_file(s3_bucket_name, s3_key, local_path)
        print(f"파일이 성공적으로 다운로드되었습니다: {local_path}")
        return pl.read_csv(local_path)
    except Exception as e:
        raise RuntimeError(f"S3에서 파일을 다운로드하는 중 에러가 발생했습니다: {e}")


def get_parquet_file(local_path, s3_bucket_name, s3_key):
    if os.path.exists(local_path):
        print(f"로컬에서 파일을 읽습니다: {local_path}")
        return pl.read_parquet(local_path)

    print(f"로컬 파일이 없습니다. S3에서 파일을 다운로드합니다: {s3_bucket_name}/{s3_key}")
    try:
        s3 = boto3.client("s3")
        s3.download_file(s3_bucket_name, s3_key, local_path)
        print(f"파일이 성공적으로 다운로드되었습니다: {local_path}")
        return pl.read_parquet(local_path)
    except Exception as e:
        raise RuntimeError(f"S3에서 파일을 다운로드하는 중 에러가 발생했습니다: {e}")

def get_rds_data(db_config, user_id):
    
    # RDS 연결
    try:
        conn = psycopg2.connect(**db_config)
        print("AWS RDS 연결 성공")
    except Exception as e:
        print("AWS RDS 연결 실패:", e)
        raise

    # SQL 쿼리 실행 및 Polars로 변환
    try:
        query = f"SELECT user_id, correct, cre_date, f_mchapter_id FROM user_testhistory WHERE user_id = '{user_id}';"  # 원하는 SQL 쿼리
        # Polars에서 직접 SQL 실행 및 DataFrame 변환
        return pl.read_database(query, connection=conn)
    except Exception as e:
        print("데이터 가져오기 실패:", e)
    finally:
        # 연결 닫기
        conn.close()
        print("RDS 연결 닫힘")