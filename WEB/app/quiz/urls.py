from django.urls import path
from . import views

urlpatterns = [
    path("<uuid:user_id>/quiz/", views.quiz_view, name="quiz"),
    path("quiz/<uuid:user_id>/result/", views.quiz_result, name="quiz_result"),
]
