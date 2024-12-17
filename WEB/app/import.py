import os
import django
import pandas as pd
from uuid import UUID
from user.models import TestHistory  # 'user'를 실제 앱 이름으로 변경
import pytz

# Django 환경 초기화
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")
django.setup()

utc = pytz.UTC


def batch_insert_records(file_path, batch_size=10000):
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
        return

    df["CreDate"] = pd.to_datetime(df["CreDate"]).dt.tz_localize(utc)
    total_records = len(df)
    print(f"총 레코드 수: {total_records}")

    for start in range(0, total_records, batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        records = []

        for _, row in batch.iterrows():
            try:
                record = TestHistory(
                    user_id=UUID(row["UserID"]),
                    m_code=row["mCode"],
                    no=row["No"],
                    quiz_code=row["QuizCode"],
                    answer=row["Answer"] if pd.notna(row["Answer"]) else None,
                    correct=row["Correct"],
                    cre_date=row["CreDate"],
                )
                records.append(record)
            except Exception as e:
                print(f"레코드 생성 실패 (index={_}): {e}")

        if records:
            try:
                TestHistory.objects.bulk_create(records)
                print(f"{start + len(records)}개 저장 완료")
            except Exception as e:
                print(f"데이터 저장 실패 (batch={start}-{end}): {e}")

    print("모든 데이터 저장 완료!")


# 실행 부분
file_path = "user/pre_01.parquet"  # 파일 경로 수정
batch_insert_records(file_path)
