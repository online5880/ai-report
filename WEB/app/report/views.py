import os
import uuid
from django.http import StreamingHttpResponse, Http404, JsonResponse
from django.shortcuts import redirect, render
from django.utils.timezone import now, make_aware, utc
from django.utils import timezone
from django.contrib import messages

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from datetime import datetime, timedelta

from report.models import LessonData, Node
from user.models import TestHistory, DailyReport
from langchain_openai import ChatOpenAI  # LangChain OpenAI 통합
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from neo4j import GraphDatabase
import json
from dotenv import load_dotenv
import re

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from django.http import HttpResponse

load_dotenv()


# ==========================================
# 사용자 입력 페이지 처리
# ==========================================
def user_input(request):
    """
    사용자 ID 입력을 처리하는 뷰 함수

    동작 방식:
    1. GET 요청: 사용자 ID 입력 폼을 표시
    2. POST 요청: 입력된 user_id를 검증 후 캘린더 페이지로 리디렉션
    """
    if request.method == "POST":
        user_id = request.POST.get("user_id")

        # UUID 형식 검증
        try:
            user_uuid = uuid.UUID(user_id)
        except (ValueError, TypeError):
            messages.error(request, "잘못된 UUID 형식입니다. 올바른 형식을 입력해주세요.")
            return render(request, "report/user_input.html")

        # DB 존재 여부 확인
        if not TestHistory.objects.filter(user_id=user_uuid).exists():
            messages.error(request, "해당 사용자 ID가 존재하지 않습니다.")
            return render(request, "report/user_input.html")

        # 검증 성공 시 리디렉션
        return redirect("calendar", user_id=user_uuid)

    return render(request, "report/user_input.html")


# ==========================================
# 학습 기록 캘린더 페이지
# ==========================================
def calendar_view(request, user_id):
    """
    사용자의 학습 기록을 캘린더 형태로 표시하는 뷰 함수

    주요 기능:
    1. 사용자의 학습 기록 날짜 조회
    2. 기존 리포트가 있는 날짜 조회
    3. 월별 데이터 구성
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

    months = list(range(1, 13))

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
    AI를 활용한 실시간 학습 리포트 생성 API
    """

    @swagger_auto_schema(
        operation_description="사용자의 학습 기록을 기반으로 AI 리포트를 생성합니다.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=["user_id"],
            properties={
                "user_id": openapi.Schema(
                    type=openapi.TYPE_STRING, description="사용자 ID"
                ),
                "date": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="조회할 날짜 (YYYY-MM-DD 형식)",
                    example="2024-01-09",
                ),
            },
        ),
        responses={
            200: openapi.Response(
                description="성공적으로 리포트가 생성됨",
                schema=openapi.Schema(
                    type=openapi.TYPE_STRING, description="스트리밍 형식의 리포트 내용"
                ),
            ),
            400: "잘못된 요청 형식",
            404: "해당 사용자와 날짜에 대한 기록이 없음",
        },
    )
    def post(self, request):
        """
        사용자의 학습 기록을 기반으로 스트리밍 리포트를 생성합니다.

        Args:
            request: Django HTTP 요청 객체

        Returns:
            StreamingHttpResponse: 스트리밍 형식의 리포트 응답
        """
        user_id = request.data.get("user_id")
        if not user_id:
            return Response({"error": "user_id는 필수 파라미터입니다."}, status=400)

        # request.data에서 date 파라미터 가져오기
        date_param = request.data.get("date")
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
            report_content = existing_report.report_content
            # mcode 패턴 정의 및 필터링
            mcode_pattern = r"T1ME\d{2}U\d{5}"
            mcodes = re.findall(mcode_pattern, report_content)

            # LessonData 모델에서 해당 mcode들을 사용하여 데이터 추출
            lesson_data = LessonData.objects.filter(mcode__in=mcodes)

            # mcode를 unique_content_nm으로 변경
            mcode_to_unique_content_nm = {
                lesson.mcode: lesson.unique_content_nm for lesson in lesson_data
            }

            # report_content에서 mcode를 unique_content_nm으로 대체
            def replace_mcode_with_unique_content_nm(match):
                mcode = match.group(0)
                return mcode_to_unique_content_nm.get(mcode, mcode)

            updated_report_content = re.sub(
                mcode_pattern, replace_mcode_with_unique_content_nm, report_content
            )

            return StreamingHttpResponse(
                updated_report_content,  # `data:` 접두어 없이 바로 출력
                content_type="text/event-stream; charset=utf-8",
            )

        # 학습 기록 조회
        histories = TestHistory.objects.filter(
            user_id=user_id, cre_date__range=(start_date, end_date)
        ).values("m_code", "quiz_code", "correct")

        if not histories.exists():
            raise Http404("해당 사용자와 날짜에 대한 기록이 없습니다.")

        # LessonData에서 mcode와 관련 데이터 매핑 생성
        lesson_data = {
            lesson["mcode"]: {
                "l_title": lesson["l_title"],
                "content_grade": lesson["content_grade"],
                "term": lesson["term"],
            }
            for lesson in LessonData.objects.filter(
                mcode__in=[history["m_code"] for history in histories]
            ).values("mcode", "l_title", "content_grade", "term")
        }

        # 학습 통계 계산
        total_attempts = histories.count()
        correct_answers = histories.filter(correct="O").count()
        incorrect_answers = total_attempts - correct_answers
        accuracy = (correct_answers / total_attempts) * 100 if total_attempts > 0 else 0

        # record_details 작성
        record_details = "\n".join(
            [
                f"모듈: {lesson_data.get(record['m_code'], {}).get('l_title', '알 수 없음')}, "
                f"학년: {lesson_data.get(record['m_code'], {}).get('content_grade', '알 수 없음')}학년, "
                f"학기: {lesson_data.get(record['m_code'], {}).get('term', '알 수 없음')}학기, "
                f"퀴즈: {record['quiz_code']}, 정답 여부: {record['correct']}"
                for record in histories
            ]
        )

        # AI 프롬프트 생성
        prompt = f"""
        사용자 학습 일일 리포트
        사용자 ID: {user_id} 님
        날짜: {date}
        요약:
        - 총 시도 횟수: {total_attempts}회
        - 정답 횟수: {correct_answers}회
        - 오답 횟수: {incorrect_answers}회
        - 정답률: {accuracy:.2f}%
        세부 기록:
        {record_details}

        ---

        # 마크다운으로 작성할 내용

        ### 요구사항
        - 초등학교 1~2학년 수준으로 쉽고 친근하게 작성 (150자 이내)
        - 학습 성과 분석
        - 개선을 위한 제안
        - 동기부여 메시지
        - 부족했던 단원(l_title)과 학년 및 학기
        - 모듈이라는 단어는 단원으로 변경

        ### 출력 예시
        # {user_id} 친구의 오늘의 학습 리포트 🌟

        ## 오늘의 성과
        * 문제 {total_attempts}개 중에서 {correct_answers}개를 맞혔어요
        * 특히 [과목명] 부분이 조금 어려웠나봐요

        ## 앞으로 이렇게 해보면 좋아요
        * [과목명] 문제는 그림을 그려가며 풀어보면 더 쉬워요
        * 어려운 문제는 선생님께 질문해보는 것도 좋아요

        ## 힘이 나는 한마디
        * 꾸준히 노력하는 {user_id} 친구가 정말 자랑스러워요!
        * 포기하지 않고 도전하는 모습이 멋져요 ⭐

        ## 더 공부하면 좋을 단원
        * [과목명1]
        * [과목명2]
        """

        # 스트리밍 리포트 반환
        return StreamingHttpResponse(
            self.stream_response(prompt, user_id, date),
            content_type="text/event-stream; charset=utf-8",
        )

    def stream_response(self, prompt, user_id, date):
        """
        AI 모델을 통한 스트리밍 응답 생성

        처리 과정:
        1. OpenAI API 연결
        2. 프롬프트 처리
        3. 응답 스트리밍
        4. 데이터베이스 저장
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


# ==========================================
# Neo4j 관련 함수 및 뷰
# ==========================================


# Neo4j 연결 설정
def clean_env_var(var):
    """
    환경 변수 문자열 정제

    처리:
    - 앞뒤 따옴표 제거
    - 공백 처리
    """
    if (
        var
        and (var.startswith('"') and var.endswith('"'))
        or (var.startswith("'") and var.endswith("'"))
    ):
        return var[1:-1]
    return var


neo4j_uri = clean_env_var(os.getenv("NEO4J_BOLT_URI"))
# neo4j_uri = "bolt://host.docker.internal:7687"
neo4j_username = clean_env_var(os.getenv("NEO4J_USERNAME"))
neo4j_password = clean_env_var(os.getenv("NEO4J_PASSWORD"))
# driver = GraphDatabase.driver("bolt://neo4j:7687", auth=(neo4j_username, "bigdata9-"))
# driver = GraphDatabase.driver(
#     "bolt://localhost:7687", auth=(neo4j_username, "bigdata9-")
# )
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))


def get_graph_data():
    """
    Neo4j 데이터베이스에서 그래프 데이터 조회

    반환 데이터:
    - nodes: 그래프의 노드 정보
    - links: 노드 간의 연결 정보

    데이터 구조:
    1. 노드: id, label, color, properties
    2. 링크: source, target, type, color, title
    """

    with driver.session() as session:
        query = "MATCH (n)-[r]->(m) RETURN n, r, m"
        result = session.run(query)
        # print("query : ", query)
        # print("result : ", result)

        nodes = []
        links = []
        node_ids = set()

        # 데이터에서 사용될 기본 색상
        default_color = "#999"

        for record in result:
            n = record["n"]
            m = record["m"]
            r = record["r"]

            # 노드 추가 (시작 노드)
            n_id = n.get("id", n.element_id)  # id 속성 또는 element_id 사용
            if n_id not in node_ids:
                nodes.append(
                    {
                        "id": n_id,
                        "label": n.get("f_mchapter_nm", "Unknown"),
                        "color": n.get("color", default_color),
                        "properties": dict(n),  # 모든 속성 포함
                    }
                )
                node_ids.add(n_id)

            # 노드 추가 (끝 노드)
            m_id = m.get("id", m.element_id)  # id 속성 또는 element_id 사용
            if m_id not in node_ids:
                nodes.append(
                    {
                        "id": m_id,
                        "label": m.get("f_mchapter_nm", "Unknown"),
                        "color": m.get("color", default_color),
                        "properties": dict(m),
                    }
                )
                node_ids.add(m_id)

            # 관계 데이터 추가
            links.append(
                {
                    "source": n_id,
                    "target": m_id,
                    "type": r.type,  # 관계 타입
                    "color": r.get("color", default_color),  # 관계 색상
                    "title": r.get("title", ""),  # 관계 속성 title 사용
                }
            )

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


class KnowledgeGraphAPI(APIView):
    """
    지식 그래프 데이터를 제공하는 REST API

    응답:
    - 성공: 그래프 데이터 (JSON)
    - 실패: 에러 메시지와 500 상태 코드
    """

    def get(self, request):
        try:
            graph_data = get_graph_data()

            return JsonResponse(graph_data)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


def graph_view(request):
    """
    그래프 시각화 페이지를 렌더링하는 뷰 함수
    """
    return render(request, "report/graph.html")


class CorrectRateAPIView(APIView):
    """
    특정 날짜의 정답률을 계산하는 API
    """

    @swagger_auto_schema(
        operation_summary="특정 날짜 정답률 계산",
        operation_description="사용자의 특정 날짜 정답률 (정답 수 / 전체 문제 수)을 계산합니다.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "user_id": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="사용자 ID (UUID 형식)",
                ),
                "target_date": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    format=openapi.FORMAT_DATE,
                    description="기준 날짜 (YYYY-MM-DD)",
                ),
            },
            required=["user_id", "target_date"],
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "correct_rate": openapi.Schema(
                        type=openapi.TYPE_NUMBER,
                        format=openapi.FORMAT_FLOAT,
                        description="정답률 (소수점 비율)",
                    ),
                    "total_questions": openapi.Schema(
                        type=openapi.TYPE_INTEGER,
                        description="총 문제 수",
                    ),
                    "correct_answers": openapi.Schema(
                        type=openapi.TYPE_INTEGER,
                        description="정답 개수",
                    ),
                },
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "error": openapi.Schema(
                        type=openapi.TYPE_STRING,
                        description="오류 메시지",
                    ),
                },
            ),
        },
    )
    def post(self, request):
        data = request.data

        # 요청 데이터 유효성 검증
        user_id = data.get("user_id")
        target_date = data.get("target_date")

        if not user_id or not target_date:
            return Response(
                {"error": "user_id와 target_date는 필수입니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # 기준 날짜의 시작과 끝 시간 계산
            target_date = datetime.strptime(target_date, "%Y-%m-%d")
            start_datetime = timezone.make_aware(
                target_date
            )  # naive datetime을 aware datetime으로 변환
            end_datetime = start_datetime + timedelta(days=1)

            # 특정 날짜의 데이터 필터링
            records = TestHistory.objects.filter(
                user_id=user_id, cre_date__range=[start_datetime, end_datetime]
            )

            total_questions = records.count()
            correct_answers = records.filter(correct="O").count()

            if total_questions == 0:
                return Response(
                    {"correct_rate": 0, "total_questions": 0, "correct_answers": 0},
                    status=status.HTTP_200_OK,
                )

            # 정답률 계산
            correct_rate = correct_answers / total_questions

            # 응답 데이터
            return Response(
                {
                    "correct_rate": correct_rate,
                    "total_questions": total_questions,
                    "correct_answers": correct_answers,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AccuracyAPIView(APIView):
    """
    기준 날짜를 포함한 5일간의 정답률 데이터를 반환하는 API
    """

    @swagger_auto_schema(
        operation_summary="5일간의 정답률 데이터 조회",
        operation_description="기준 날짜를 포함하여 총 5일간의 정답률 데이터를 반환합니다.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "user_id": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="사용자 ID (UUID 형식)",
                ),
                "target_date": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    format=openapi.FORMAT_DATE,
                    description="기준 날짜 (YYYY-MM-DD 형식)",
                ),
            },
            required=["user_id", "target_date"],
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "labels": openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_STRING),
                        description="5일간의 요일 레이블 (['월', '화', '수', '목', '금'])",
                    ),
                    "data": openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_NUMBER),
                        description="5일간의 정답률 데이터 (소수점 포함 비율, [%])",
                    ),
                },
                description="정답률 데이터",
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "error": openapi.Schema(
                        type=openapi.TYPE_STRING,
                        description="에러 메시지",
                    ),
                },
                description="잘못된 요청",
            ),
        },
    )
    def post(self, request):
        data = request.data
        user_id = data.get("user_id")
        target_date = data.get("target_date")

        if not user_id or not target_date:
            return Response(
                {"error": "user_id와 target_date는 필수입니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # 기준 날짜 설정
            target_date = datetime.strptime(target_date, "%Y-%m-%d")
            labels = [
                (target_date + timedelta(days=i)).strftime("%a") for i in range(-2, 3)
            ]
            accuracy_data = []

            for i in range(-2, 3):  # 기준 날짜 포함 총 5일
                day = target_date + timedelta(days=i)
                start_datetime = datetime.combine(day, datetime.min.time()).replace(
                    tzinfo=utc
                )
                end_datetime = datetime.combine(day, datetime.max.time()).replace(
                    tzinfo=utc
                )

                # 데이터 필터링
                records = TestHistory.objects.filter(
                    user_id=user_id, cre_date__range=[start_datetime, end_datetime]
                )

                total_questions = records.count()
                correct_answers = records.filter(correct__iexact="O").count()

                # 정답률 계산
                accuracy = (
                    (correct_answers / total_questions) * 100
                    if total_questions > 0
                    else 0
                )
                accuracy_data.append(round(accuracy, 2))

            return Response(
                {"labels": labels, "data": accuracy_data}, status=status.HTTP_200_OK
            )

        except Exception as e:
            print(f"[ERROR] Exception: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# Neo4j 연결 설정
graph = Neo4jGraph(
    # url="bolt://host.docker.internal:7687",
    url=neo4j_uri,
    username=neo4j_username,
    password=neo4j_password,
)


class GraphDataAPIView(APIView):
    """
    Neo4j 그래프 데이터를 가져오는 API
    """

    @swagger_auto_schema(
        operation_summary="중단원 코드로 지식 그래프(NEO4J) 데이터 조회",
        operation_description="POST 요청으로 중단원 코드를 넣으면 Cypher 쿼리를 실행하고 그래프 데이터를 반환합니다.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "message": openapi.Schema(
                    type=openapi.TYPE_STRING,
                    description="질문 또는 요청 메시지",
                )
            },
            required=["message"],
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "query": openapi.Schema(
                        type=openapi.TYPE_STRING, description="실행된 Cypher 쿼리"
                    ),
                    "nodes": openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_OBJECT),
                        description="노드 데이터",
                    ),
                    "links": openapi.Schema(
                        type=openapi.TYPE_ARRAY,
                        items=openapi.Schema(type=openapi.TYPE_OBJECT),
                        description="링크 데이터",
                    ),
                },
                description="그래프 데이터 응답",
            ),
            400: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "error": openapi.Schema(
                        type=openapi.TYPE_STRING, description="에러 메시지"
                    )
                },
                description="잘못된 요청",
            ),
        },
    )
    def post(self, request):
        """
        POST 요청으로 중단원 코드를 넣으면 Cypher 쿼리를 실행하고 그래프 데이터를 반환합니다.
        """

        # 요청 데이터에서 메시지 가져오기
        message = request.data.get("message", "").strip()

        if not message:
            return Response(
                {"error": "message 필드는 필수입니다."}, status=status.HTTP_400_BAD_REQUEST
            )

        # LangChain 설정
        llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-11-20")
        chain = GraphCypherQAChain.from_llm(
            cypher_llm=llm,
            qa_llm=llm,
            validate_cypher=True,
            graph=graph,
            verbose=True,
            return_intermediate_steps=True,
            return_direct=True,
            allow_dangerous_requests=True,
        )

        try:
            # Cypher 쿼리 실행
            result = chain.invoke(message)

            # 결과 구조 확인
            if isinstance(result, dict) and "query" in result and "result" in result:
                query = result["query"]
                raw_data = result["result"]
            else:
                raise ValueError("Unexpected result format from LangChain.")

            # 노드와 링크 데이터 처리
            nodes = []
            links = []
            node_ids = set()

            # 데이터 처리
            for record in raw_data:
                node = record.get("n")
                if not node:
                    continue

                node_id = node.get("id")
                if node_id and node_id not in node_ids:
                    # 노드 추가
                    nodes.append(
                        {
                            "id": node_id,
                            "label": node.get("f_mchapter_nm", "Unknown"),
                            "properties": node,
                            "color": node.get("color", "#000000"),  # 기본값 검정색
                        }
                    )
                    node_ids.add(node_id)

            # 노드 정렬 (f_lchapter_id 또는 id를 기준으로 정렬)
            nodes.sort(key=lambda x: (x["properties"].get("f_lchapter_id"), x["id"]))

            # i+1 연결 생성 (제한 조건 추가)
            for i in range(len(nodes) - 1):
                if nodes[i]["properties"].get("f_lchapter_id") == nodes[i + 1][
                    "properties"
                ].get("f_lchapter_id"):
                    links.append(
                        {
                            "source": nodes[i]["id"],
                            "target": nodes[i + 1]["id"],
                            "type": "RELATES_TO",
                        }
                    )

            # 추가 연결 조건 (area 및 f_lchapter_nm 기준)
            for source_node in nodes:
                for target_node in nodes:
                    if (
                        source_node["id"] != target_node["id"]
                        and source_node["properties"].get("area")
                        == target_node["properties"].get("area")
                        and (
                            source_node["properties"]
                            .get("f_lchapter_nm")
                            .startswith("5.")
                            or target_node["properties"]
                            .get("f_lchapter_nm")
                            .startswith("5.")
                        )
                    ):
                        links.append(
                            {
                                "source": source_node["id"],
                                "target": target_node["id"],
                                "type": "RELATES_TO",
                            }
                        )

            # 중복된 링크 제거
            unique_links = list(
                {(link["source"], link["target"]): link for link in links}.values()
            )
            # print("query : ", query)
            # print("unique_links : ", unique_links)
            return Response(
                {"query": query, "nodes": nodes, "links": unique_links},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


def get_pyvis_html(request, user_id):
    file_path = os.path.join("static", "reconstructed_graph.html")  # 정적 파일 경로
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HttpResponse(html_content, content_type="text/html")
    else:
        return HttpResponse("파일을 찾을 수 없습니다.", status=404)


class NodeDetailView(APIView):
    """
    특정 node_id로 f_mchapter_nm을 반환하는 API
    """

    def get(self, request, node_id):
        try:
            # node_id로 Node 객체 검색
            node = Node.objects.get(node_id=node_id)
            return Response(
                {"node_id": node.node_id, "f_mchapter_nm": node.f_mchapter_nm},
                status=status.HTTP_200_OK,
            )
        except Node.DoesNotExist:
            return Response(
                {"error": "Node not found"}, status=status.HTTP_404_NOT_FOUND
            )
