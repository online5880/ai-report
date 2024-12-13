from django.shortcuts import redirect, render
from user.models import TestHistory

# Create your views here.


def user_input(request):
    if request.method == "POST":
        user_id = request.POST.get("user_id")
        return redirect("calendar", user_id=user_id)
    return render(request, "report/user_input.html")


def calendar_view(request, user_id):
    histories = TestHistory.objects.filter(user_id=user_id)
    dates = list(histories.values_list("cre_date", flat=True).distinct())  # 리스트로 변환
    print(dates)
    return render(request, "report/calendar.html", {"user_id": user_id, "dates": dates})
