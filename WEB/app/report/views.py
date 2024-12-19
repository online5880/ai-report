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

from neo4j import GraphDatabase
import json
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# 사용자 입력 페이지 처리
# ==========================================
def user_input(request):
    """
    사용자 입력 페이지
    - POST 요청 시 입력받은 user_id를 사용하여 캘린더 페이지로 리디렉션.
    - GET 요청 시 입력 폼 렌더링.

    Args:
        request: Django HTTP 요청 객체.

    Returns:
        - POST: 캘린더 페이지로 리디렉션.
        - GET: 사용자 입력 페이지 렌더링.
    """
    if request.method == "POST":
        user_id = request.POST.get("user_id")
        return redirect("calendar", user_id=user_id)
    return render(request, "report/user_input.html")


# ==========================================
# 학습 기록 캘린더 페이지
# ==========================================
def calendar_view(request, user_id):
    """
    특정 사용자의 학습 기록을 캘린더로 표시.
    - 사용자가 학습한 날짜 리스트를 반환.

    Args:
        request: Django HTTP 요청 객체.
        user_id: 학습 기록을 조회할 사용자 ID.

    Returns:
        캘린더 페이지 렌더링.
    """
    # 학습 기록 날짜 추출
    dates = TestHistory.objects.filter(user_id=user_id).values_list(
        "cre_date", flat=True
    )
    dates = [date.date() for date in dates]  # 날짜만 추출

    # 리포트가 존재하는 날짜들 조회
    report_dates = DailyReport.objects.filter(user_id=user_id).values_list(
        "date", flat=True
    )
    report_dates = [date.strftime("%Y-%m-%d") for date in report_dates]  # 문자열 형식으로 변환

    months = range(1, 13)

    return render(
        request,
        "report/calendar.html",
        {
            "user_id": user_id,
            "dates": list(set(dates)),
            "months": months,
            "report_dates": report_dates,
        },
    )


# ==========================================
# 특정 날짜의 학습 리포트 보기
# ==========================================
def view_report(request, user_id, date):
    """
    특정 사용자의 특정 날짜 학습 리포트를 조회 및 렌더링.

    Args:
        request: Django HTTP 요청 객체.
        user_id: 사용자 ID.
        date: 학습 기록 날짜 (YYYY-MM-DD 형식).

    Returns:
        - 리포트 페이지 렌더링.
        - 날짜 형식이 잘못되었거나 리포트가 없으면 404 에러 반환.
    """
    try:
        # 날짜 형식 확인
        report_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise Http404("잘못된 날짜 형식입니다.")

    # 리포트 조회
    try:
        report = DailyReport.objects.get(user_id=user_id, date=report_date)
    except DailyReport.DoesNotExist:
        report = None

    return render(
        request,
        "report/view_report.html",
        {
            "user_id": user_id,
            "report": report,
        },
    )


# ==========================================
# AI 스트리밍 일일 리포트 API
# ==========================================
class StreamingDailyReportAPI(APIView):
    """
    AI 기반 스트리밍 학습 리포트 생성 API.
    - 사용자의 학습 기록을 기반으로 AI 모델이 스트리밍 방식으로 리포트를 생성.
    """

    def stream_response(self, prompt, user_id, date):
        """
        AI 스트리밍 응답 생성.
        - 스트리밍된 데이터를 한 번에 처리하지 않고 순차적으로 반환.

        Args:
            prompt: AI 모델에 전달할 프롬프트.
            user_id: 사용자 ID.
            date: 학습 기록 날짜.

        Yields:
            AI 모델이 생성한 리포트 텍스트.
        """
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
            streaming=True,
        )

        chain = RunnablePassthrough() | llm | StrOutputParser()

        report_content = ""
        try:
            # 스트리밍 데이터를 하나씩 반환
            for chunk in chain.stream(prompt):
                report_content += chunk
                cleaned_chunk = chunk.replace("data: ", "")  # `data:` 접두어 제거
                yield cleaned_chunk

            # 리포트를 데이터베이스에 저장
            if report_content:
                existing_report = DailyReport.objects.filter(
                    user_id=user_id, date=date
                ).first()
                if not existing_report:
                    DailyReport.objects.create(
                        user_id=user_id, date=date, report_content=report_content
                    )

        except Exception as e:
            # 예외 처리: 스트리밍 도중 오류가 발생한 경우
            yield f"data: 오류 발생: {str(e)}\n\n"

    def get(self, request, user_id):
        """
        사용자의 학습 기록을 기반으로 스트리밍 리포트를 생성합니다.

        Args:
            request: Django HTTP 요청 객체.
            user_id: 사용자 ID.

        Returns:
            - 기존 리포트가 있으면 해당 리포트를 반환.
            - 없으면 새로운 리포트를 생성하여 반환.
        """
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

        # 이미 해당 날짜에 리포트가 존재하는지 확인
        existing_report = DailyReport.objects.filter(user_id=user_id, date=date).first()
        if existing_report:
            # 리포트가 존재하면 기존 리포트를 스트리밍 형식으로 반환
            return StreamingHttpResponse(
                existing_report.report_content,  # `data:` 접두어 없이 바로 출력
                content_type="text/event-stream; charset=utf-8",
            )

        # 학습 기록 조회
        histories = TestHistory.objects.filter(
            user_id=user_id, cre_date__range=(start_date, end_date)
        ).values("m_code", "quiz_code", "answer", "correct")

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
                f"모듈: {record['m_code']}, 퀴즈: {record['quiz_code']}, "
                f"답변: {record['answer']}, 정답 여부: {record['correct']}"
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

        # 필수
        초등학생 1,2학년 수준으로 작성하세요.!
        """

        # 스트리밍 리포트 반환
        return StreamingHttpResponse(
            self.stream_response(prompt, user_id, date),
            content_type="text/event-stream; charset=utf-8",
        )


# ==========================================
# Neo4j 관련 함수 및 뷰
# ==========================================

# Neo4j 연결 설정
uri = os.getenv("NEO4J_BOLT_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(uri, auth=(username, password))


def get_graph_data():
    """
    Neo4j에서 그래프 데이터를 조회합니다.
    - 노드와 관계 데이터를 가져와 JSON 형식으로 반환.

    Returns:
        dict: 노드와 관계 데이터를 포함하는 JSON 객체.
    """
    with driver.session() as session:
        query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50"
        result = session.run(query)

        nodes = []
        links = []

        for record in result:
            n = record["n"]
            m = record["m"]
            r = record["r"]

            # 노드 데이터 추가
            if n.id not in [node["id"] for node in nodes]:
                nodes.append(
                    {
                        "id": n.id,
                        "label": list(n.labels)[0] if n.labels else None,
                        "name": n.get("id", "Unknown"),
                        "properties": dict(n),
                    }
                )
            if m.id not in [node["id"] for node in nodes]:
                nodes.append(
                    {
                        "id": m.id,
                        "label": list(m.labels)[0] if m.labels else None,
                        "name": m.get("id", "Unknown"),
                        "properties": dict(m),
                    }
                )

            # 관계 데이터 추가
            links.append({"source": n.id, "target": m.id, "type": r.type})

        return {"nodes": nodes, "links": links}


def neo4j_view(request):
    """
    Neo4j 데이터를 템플릿으로 렌더링합니다.

    Args:
        request: Django HTTP 요청 객체.

    Returns:
        HTML 템플릿 렌더링.
    """
    graph_data = get_graph_data()
    return render(
        request, "report/neo4j_page.html", {"graph_data": json.dumps(graph_data)}
    )
