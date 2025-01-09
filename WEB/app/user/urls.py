from django.urls import path
from user.views import TestHistoryAPIView  # 'user'를 실제 앱 이름으로 변경

urlpatterns = [
    path(
        "api/testhistory/<str:user_id>/",
        TestHistoryAPIView.as_view(),
        name="testhistory_api",
    ),
]
