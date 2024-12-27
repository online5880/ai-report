from django.urls import path
from . import views

urlpatterns = [
    path("", views.user_input, name="user_input"),
    path("calendar/<str:user_id>/", views.calendar_view, name="calendar"),
    path("report/<str:user_id>/<str:date>/", views.view_report, name="view_report"),
    path(
        "api/streaming-daily-report/<str:user_id>/",
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
]
