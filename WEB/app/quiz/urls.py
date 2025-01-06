from django.urls import path
from . import views

urlpatterns = [
    path('<str:user_id>/quiz/', views.quiz_view, name='quiz_page'),
]