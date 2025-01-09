from rest_framework import serializers
from user.models import TestHistory  # 'user'를 실제 앱 이름으로 변경


class TestHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = TestHistory
        fields = "__all__"
