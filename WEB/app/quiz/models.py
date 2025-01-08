from django.db import models

class Question(models.Model):
    text = models.CharField(max_length=255)
    f_mchapter_id = models.BigIntegerField(db_index=True, default=14201897)  # 중단원 코드 추가

    def __str__(self):
        return self.text

class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='choices')
    text = models.CharField(max_length=255)
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return self.text
