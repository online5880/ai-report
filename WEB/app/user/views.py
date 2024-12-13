from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import TestHistory
from .serializers import TestHistorySerializer


# 학습 결과 기록 API
class TestHistoryAPIView(APIView):
    def get(self, request, user_id):

        # user_id에 해당하는 데이터 가져오기
        histories = TestHistory.objects.filter(user_id=user_id)

        # serializer로 데이터 변환
        serializer = TestHistorySerializer(histories, many=True)

        return Response(serializer.data, status=status.HTTP_200_OK)
