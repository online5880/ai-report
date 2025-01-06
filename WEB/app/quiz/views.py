from django.shortcuts import render
from django.http import JsonResponse
from django.utils.timezone import now
from user.models import TestHistory
from .models import Question

def quiz_view(request, user_id):
    if request.method == 'GET':
        # 문제를 랜덤으로 가져오기 (예: 10개)
        questions = Question.objects.all()[:10]
        return render(request, 'quiz/quiz.html', {'questions': questions, 'user_id': user_id})

    elif request.method == 'POST':
        data = request.POST
        score = 0
        total = 0

        for key, value in data.items():
            if key.startswith("question_"):  # 문제의 키 형식
                question_id = key.split("_")[1]
                question = Question.objects.get(id=question_id)
                selected_choice = question.choices.get(id=value)
                is_correct = 'O' if selected_choice.is_correct else 'X'

                # DB에 기록 저장
                TestHistory.objects.create(
                    user_id=user_id,
                    m_code='quiz_module',
                    no=question_id,
                    quiz_code=selected_choice.id,
                    answer=selected_choice.text,
                    correct=is_correct,
                    cre_date=now(),
                )

                total += 1
                if selected_choice.is_correct:
                    score += 1

        return JsonResponse({'message': '결과 저장 완료', 'score': score, 'total': total})