from django.shortcuts import render
from django.http import JsonResponse
from django.utils.timezone import now
from user.models import TestHistory
from .models import Question
import requests
import json


def quiz_view(request, user_id):
    if request.method == "GET":
        questions = Question.objects.all()[:10]
        return render(
            request, "quiz/quiz.html", {"questions": questions, "user_id": user_id}
        )

    elif request.method == "POST":
        data = request.POST
        score = 0
        total = 0
        skill_list = []
        correct_list = []

        # 문제 결과 수집
        for key, value in data.items():
            if key.startswith("question_"):
                question_id = key.split("_")[1]
                question = Question.objects.get(id=question_id)
                selected_choice = question.choices.get(id=value)

                # 리스트에 추가
                skill_list.append(question.f_mchapter_id)
                correct_list.append(0 if selected_choice.is_correct else 1)

                print(f"question_{question_id}: {selected_choice.is_correct}")

                TestHistory.objects.create(
                    user_id=user_id,
                    m_code="quiz_module",
                    quiz_code=selected_choice.id,
                    correct="O" if selected_choice.is_correct else "X",
                    cre_date=now(),
                    f_mchapter_id=question.f_mchapter_id,
                )

                total += 1
                if selected_choice.is_correct:
                    score += 1

                print("correct_list:", correct_list)
                print("skill_list:", skill_list)

        # GKT API 호출
        try:
            gkt_data = {
                "user_id": str(user_id),
                "skill_list": skill_list,
                "correct_list": correct_list,
            }

            print("gkt_data:", gkt_data)

            gkt_response = requests.post(
                "http://mane.my/api/gkt",
                headers={"Content-Type": "application/json"},
                json=gkt_data,
            )

            if gkt_response.ok:
                predictions = gkt_response.json()
                # predictions를 세션에 저장하여 결과 페이지에서 사용
                request.session["predictions"] = predictions
            else:
                print(f"GKT API 호출 실패: {gkt_response.status_code}")

        except Exception as e:
            print(f"GKT API 호출 중 오류 발생: {str(e)}")

        return JsonResponse(
            {
                "message": "결과 저장 완료",
                "score": score,
                "total": total,
                "redirect_url": f"/quiz/{user_id}/result/",
            }
        )


def quiz_result(request, user_id):
    score = request.GET.get("score", 0)
    total = request.GET.get("total", 0)
    predictions = request.session.get("predictions", {})
    print("=" * 50)
    print("predictions:", predictions)
    print("=" * 50)
    return render(
        request,
        "quiz/knowledge_graph.html",
        {
            "user_id": user_id,
            "score": score,
            "total": total,
            "predictions": json.dumps(predictions),
        },
    )
