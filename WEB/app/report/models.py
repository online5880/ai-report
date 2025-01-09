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


class Node(models.Model):
    node_id = models.IntegerField(primary_key=True, db_index=True)
    f_subject_id = models.IntegerField(db_index=True)
    f_mchapter_nm = models.CharField(max_length=255, db_index=True)
    f_lchapter_id = models.IntegerField(db_index=True)
    f_lchapter_nm = models.CharField(max_length=255, db_index=True)
    f_schapter_id = models.IntegerField(db_index=True)
    f_schapter_nm = models.CharField(max_length=255, db_index=True)
    f_tchapter_id = models.IntegerField(db_index=True)
    f_tchapter_nm = models.CharField(max_length=255, db_index=True)
    area = models.CharField(max_length=100, db_index=True)

    def __str__(self):
        return f"Node {self.node_id}: {self.f_mchapter_nm}"
