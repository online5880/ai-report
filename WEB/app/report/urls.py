from django.urls import path
from . import views
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions

# Swagger 문서 설정
schema_view = get_schema_view(
    openapi.Info(
        title="학습 리포트 API",
        default_version="v1",
        description="AI 기반 학습 분석 및 리포트 생성 API",
        contact=openapi.Contact(email="gjtjqkr5880@gamil.com"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Swagger UI 경로
    path(
        "swagger<format>/", schema_view.without_ui(cache_timeout=0), name="schema-json"
    ),
    path(
        "swagger/",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema-swagger-ui",
    ),
    path("redoc/", schema_view.with_ui("redoc", cache_timeout=0), name="schema-redoc"),
    # 기존 URL 패턴
    path("", views.user_input, name="user_input"),
    path("calendar/<str:user_id>/", views.calendar_view, name="calendar"),
    path("report/<str:user_id>/<str:date>/", views.view_report, name="view_report"),
    path(
        "api/streaming-daily-report/",
        views.StreamingDailyReportAPI.as_view(),
        name="streaming-daily-report",
    ),
    path(
        "api/knowledge-graph/",
        views.KnowledgeGraphAPI.as_view(),
        name="knowledge-graph",
    ),
    path("neo4j/", views.neo4j_view, name="neo4j_view"),
    path("graph/", views.graph_view, name="graph"),
    path(
        "api/correct-rate/", views.CorrectRateAPIView.as_view(), name="correct-rate-api"
    ),
    path("api/accuracy/", views.AccuracyAPIView.as_view(), name="accuracy-api"),
    path("api/graph-data/", views.GraphDataAPIView.as_view(), name="graph-data"),
]
