from django.db import models
import uuid


class TestHistory(models.Model):
    """
    TestHistory 모델은 사용자의 학습 기록 데이터를 저장합니다.

    필드:
        user_id (UUIDField): 사용자 ID를 저장하는 필드. UUID 형식을 사용합니다.
        m_code (CharField): 학습 코드 또는 모듈 코드를 저장합니다. 최대 50자.
        no (IntegerField): 학습 단계 또는 순서를 나타내는 정수 값.
        quiz_code (BigIntegerField): 퀴즈 식별 번호를 저장하는 필드. 큰 정수도 지원합니다.
        answer (TextField): 사용자가 입력한 답변을 저장합니다.
        correct (CharField): 답변의 정오 여부를 저장합니다. ('O' 또는 'X'로 저장)
        cre_date (DateTimeField): 학습 또는 기록 생성 날짜 및 시간을 저장합니다.

    용도:
        - 사용자의 학습 기록을 데이터베이스에 저장 및 관리.
        - 학습 데이터를 기반으로 통계 및 분석 수행.

    예제:
        TestHistory.objects.create(
            user_id=uuid.UUID('4f672b16-c7d3-40f4-b6d7-027f1bb15331'),
            m_code='T0EE20U01121',
            no=1,
            quiz_code=30126926,
            answer='singer',
            correct='O',
            cre_date='2024-01-01T19:45:00'
        )
    """

    user_id = models.UUIDField(default=uuid.uuid4, db_index=True)  # UUID
    m_code = models.CharField(max_length=50, db_index=True)  # 문자열 (mCode)
    quiz_code = models.BigIntegerField(db_index=True)  # 정수형 (QuizCode), 큰 정수 처리
    correct = models.CharField(max_length=1, db_index=True)  # 한 글자 문자열 (Correct)
    cre_date = models.DateTimeField(db_index=True)  # 날짜/시간 (CreDate)
    f_mchapter_id = models.BigIntegerField(db_index=True, default=14201897)  # 문자열 (F_MChapter)  

    def __str__(self):
        return f"{self.user_id} - {self.m_code}"


class DailyReport(models.Model):
    user_id = models.UUIDField(default=uuid.uuid4, db_index=True)
    date = models.DateField(db_index=True)  # 날짜를 기준으로 리포트 저장
    report_content = models.TextField()  # 리포트 내용 저장
    created_at = models.DateTimeField(auto_now_add=True)  # 생성일시

    class Meta:
        unique_together = ("user_id", "date")  # 동일한 날짜에 대해 중복 저장 방지

    def __str__(self):
        return f"리포트 ({self.date}) - {self.user_id}"
