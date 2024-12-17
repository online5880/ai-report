import os
from django.http import StreamingHttpResponse, Http404
from django.shortcuts import redirect, render
from django.utils.timezone import now, make_aware, utc

from rest_framework.views import APIView
from rest_framework.response import Response

from datetime import datetime
from user.models import TestHistory, DailyReport
from langchain_openai import ChatOpenAI  # LangChain OpenAI 통합
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def user_input(request):
    if request.method == "POST":
        user_id = request.POST.get("user_id")
        return redirect("calendar", user_id=user_id)
    return render(request, "report/user_input.html")


def calendar_view(request, user_id):
    # 특정 사용자의 학습 기록 날짜 가져오기
    dates = TestHistory.objects.filter(user_id=user_id).values_list(
        "cre_date", flat=True
    )
    dates = [date.date() for date in dates]  # 날짜만 추출
    months = range(1, 13)  # 1월~12월
    return render(
        request,
        "report/calendar.html",
        {"user_id": user_id, "dates": list(set(dates)), "months": months},
    )


def view_report(request, user_id, date):
    try:
        # date를 datetime 객체로 변환
        report_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise Http404("잘못된 날짜 형식입니다.")

    # 해당 날짜와 user_id에 맞는 리포트를 조회
    try:
        report = DailyReport.objects.get(user_id=user_id, date=report_date)
    except DailyReport.DoesNotExist:
        report = None  # 리포트가 없으면 None으로 처리

    return render(
        request,
        "report/view_report.html",
        {
            "user_id": user_id,
            "report": report,
        },
    )


class StreamingDailyReportAPI(APIView):
    """
    사용자의 학습 기록을 기반으로 AI 스트리밍 일일 리포트를 생성하는 API
    """

    def stream_response(self, prompt, user_id, date):
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            streaming=True,
        )

        chain = RunnablePassthrough() | llm | StrOutputParser()

        report_content = ""

        for chunk in chain.stream(prompt):
            report_content += chunk
            print(report_content)
            yield chunk

        if report_content:
            existing_report = DailyReport.objects.filter(
                user_id=user_id, date=date
            ).first()
            if not existing_report:
                DailyReport.objects.create(
                    user_id=user_id, date=date, report_content=report_content
                )

    def get(self, request, user_id):
        """
        특정 사용자의 학습 기록을 바탕으로 리포트를 생성합니다.
        """
        # 날짜 파라미터 처리
        date_param = request.query_params.get("date")
        try:
            if date_param:
                date = datetime.strptime(date_param, "%Y-%m-%d").date()
            else:
                date = now().date()

            # UTC 시간대로 필터링 범위 설정
            start_date = make_aware(
                datetime.combine(date, datetime.min.time()), timezone=utc
            )
            end_date = make_aware(
                datetime.combine(date, datetime.max.time()), timezone=utc
            )

        except ValueError:
            return Response(
                {"error": "잘못된 날짜 형식입니다. YYYY-MM-DD 형식을 사용하세요."},
                status=400,
            )

        # 학습 기록 조회
        histories = TestHistory.objects.filter(
            user_id=user_id, cre_date__range=(start_date, end_date)
        )

        print("SQL Query:", histories.query)  # SQL 쿼리 디버깅

        if not histories.exists():
            raise Http404("해당 사용자와 날짜에 대한 기록이 없습니다.")

        # 학습 통계 계산
        total_attempts = histories.count()
        correct_answers = histories.filter(correct="O").count()
        incorrect_answers = total_attempts - correct_answers
        accuracy = (correct_answers / total_attempts) * 100 if total_attempts > 0 else 0

        record_details = "\n".join(
            [
                f"모듈: {record.m_code}, 퀴즈: {record.quiz_code}, "
                f"답변: {record.answer}, 정답 여부: {record.correct}"
                for record in histories
            ]
        )

        # AI 프롬프트 생성
        prompt = f"""
        사용자 학습 일일 리포트
        사용자 ID: {user_id}
        날짜: {date}
        요약:
        - 총 시도 횟수: {total_attempts}회
        - 정답 횟수: {correct_answers}회
        - 오답 횟수: {incorrect_answers}회
        - 정답률: {accuracy:.2f}%
        세부 기록:
        {record_details}
        위 데이터를 기반으로 다음 내용을 포함한 일일 리포트를 작성하세요:
        - 학습 성과 분석
        - 개선을 위한 제안
        - 동기부여 메시지
        """

        return StreamingHttpResponse(
            self.stream_response(prompt, user_id, date),
            content_type="text/event-stream; charset=utf-8",
        )
