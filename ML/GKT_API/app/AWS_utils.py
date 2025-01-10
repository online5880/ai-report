import os
import boto3
import polars as pl
import psycopg2

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


def save_predictions_to_db(db_config, user_id, predictions):
    """
    gkt_predictions 테이블에 예측 결과 저장 (동일한 user_id의 기존 데이터 삭제 후 삽입)
    :param db_config: 데이터베이스 설정 (dict)
    :param user_id: 사용자 ID
    :param predictions: [{"skill1": 0.85}, {"skill2": 0.90}, ...]
    """
    try:
        # RDS 연결
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        print("RDS 연결 성공")

        # 기존 데이터 삭제
        delete_query = "DELETE FROM gkt_predictions WHERE user_id = %s;"
        cur.execute(delete_query, (user_id,))
        print(f"user_id={user_id}의 기존 데이터 삭제 완료.")

        # 새로운 데이터 삽입
        insert_query = """
        INSERT INTO gkt_predictions (user_id, skill, predicted_probability)
        VALUES (%s, %s, %s);
        """
        for pred in predictions:
            for skill, prob in pred.items():  # 예측 결과에서 skill과 확률 분리
                cur.execute(insert_query, (user_id, skill, prob))

        # 변경 사항 커밋
        conn.commit()
        print("예측 결과가 성공적으로 저장되었습니다.")
    except Exception as e:
        print("예측 결과 저장 실패:", e)
        raise
    finally:
        # 연결 닫기
        if conn:
            cur.close()
            conn.close()
            print("RDS 연결 닫힘")


def save_confusion_results_to_db(db_config, user_id, confusion_results):
    """
    gkt_confusion_matrix 테이블에 혼동 행렬 데이터를 저장 (동일한 user_id의 기존 데이터 삭제 후 삽입)
    :param db_config: 데이터베이스 설정 (dict)
    :param user_id: 사용자 ID
    :param confusion_results: 
        [{"skill": "skill1", "predicted_probability": 0.85, "predicted_result": 1, "actual_result": 1, "analysis": "개념 확립"}, ...]
    """
    try:
        # RDS 연결
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        print("RDS 연결 성공")

        # 기존 데이터 삭제
        delete_query = "DELETE FROM gkt_confusion_matrix WHERE user_id = %s;"
        cur.execute(delete_query, (user_id,))
        print(f"user_id={user_id}의 기존 데이터 삭제 완료.")

        # 새로운 데이터 삽입
        insert_query = """
        INSERT INTO gkt_confusion_matrix (user_id, skill, predicted_probability, predicted_result, actual_result, analysis)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        for result in confusion_results:
            cur.execute(
                insert_query,
                (
                    user_id,
                    result["skill"],
                    result["predicted_probability"],
                    result["predicted_result"],
                    result["actual_result"],
                    result["analysis"],
                ),
            )

        # 변경 사항 커밋
        conn.commit()
        print("혼동 행렬 결과가 성공적으로 저장되었습니다.")
    except Exception as e:
        print("혼동 행렬 저장 실패:", e)
        raise
    finally:
        # 연결 닫기
        if conn:
            cur.close()
            conn.close()
            print("RDS 연결 닫힘")