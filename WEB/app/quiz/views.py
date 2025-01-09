from django.shortcuts import render
from django.http import JsonResponse
from django.utils.timezone import now
from report.models import Node
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

        # GKT API 호출
        try:
            gkt_data = {
                "user_id": str(user_id),
                "skill_list": skill_list,
                "correct_list": correct_list,
            }

            gkt_response = requests.post(
                "http://mane.my/api/gkt/confusion-matrix",
                headers={"Content-Type": "application/json"},
                json=gkt_data,
            )

            if gkt_response.ok:
                predictions = gkt_response.json()

                print(predictions)

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

    # predictions가 None이 아니고 실제 데이터가 있는지 확인
    if (
        predictions
        and isinstance(predictions, dict)
        and "confusion_matrix" in predictions
    ):
        confusion_matrix = predictions["confusion_matrix"]

        # confusion_matrix가 리스트인지 확인
        if isinstance(confusion_matrix, list):
            for entry in confusion_matrix:
                skill_id = entry.get("skill")
                if skill_id:
                    try:
                        node = Node.objects.get(node_id=skill_id)
                        entry["skill_name"] = node.f_mchapter_nm
                    except Node.DoesNotExist:
                        entry["skill_name"] = "Unknown Skill"
                        print(f"Node not found for skill_id: {skill_id}")  # 디버깅용

    # 디버깅을 위한 출력 추가
    print("Modified predictions:", predictions)

    context = {
        "user_id": user_id,
        "score": score,
        "total": total,
        "predictions": json.dumps(
            predictions, ensure_ascii=False
        ),  # 한글 지원을 위해 ensure_ascii=False 추가
        "predictions_raw": predictions,  # JSON 직렬화 전 데이터도 전달
    }

    return render(request, "quiz/knowledge_graph.html", context)
