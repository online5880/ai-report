from django.db import models


class LessonData(models.Model):
    mcode = models.CharField(max_length=255)
    l_title = models.CharField(max_length=255)
    unique_content_nm = models.CharField(max_length=255)
    leccode = models.CharField(max_length=255)
    u_title = models.CharField(max_length=255)
    content_grade = models.IntegerField()
    term = models.IntegerField()

    def __str__(self):
        return f"{self.mcode} - {self.l_title}"
