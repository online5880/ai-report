from django.urls import path
from . import views

urlpatterns = [
    path("", views.user_input, name="user_input"),
    path("calendar/<str:user_id>/", views.calendar_view, name="calendar"),
    path(
        "api/streaming-daily-report/<str:user_id>/",
        views.StreamingDailyReportAPI.as_view(),
        name="streaming-daily-report",
    ),
    path("report/<uuid:user_id>/<str:date>/", views.view_report, name="view_report"),
]
