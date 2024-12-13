from django.urls import path
from . import views

urlpatterns = [
    path("", views.user_input, name="user_input"),
    path("calendar/<str:user_id>/", views.calendar_view, name="calendar"),
]
