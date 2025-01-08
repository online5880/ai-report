from django.contrib import admin
from .models import Question, Choice

class ChoiceInline(admin.TabularInline):
    model = Choice
    extra = 4  # 기본으로 4개의 선택지를 표시
    min_num = 2  # 최소 2개의 선택지
    max_num = 10  # 최대 10개의 선택지
    fields = ('text', 'is_correct')  # 보기 텍스트와 정답 여부만 표시

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('text', 'f_mchapter_id')  # 문제와 중단원 코드를 목록에 표시
    search_fields = ('text',)  # 문제를 검색 가능
    list_filter = ('f_mchapter_id',)  # 중단원 코드별 필터링 가능
    inlines = [ChoiceInline]  # 보기 관리 기능 추가
