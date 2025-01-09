from django.test import TestCase
from app.models import YourModel
from django.urls import reverse


class YourModelTestCase(TestCase):
    def setUp(self):
        YourModel.objects.create(field1="value1", field2="value2")

    def test_model_str(self):
        obj = YourModel.objects.get(field1="value1")
        self.assertEqual(str(obj), "expected_string_representation")


class YourViewTestCase(TestCase):
    def test_view_status_code(self):
        response = self.client.get(reverse("your_view_name"))
        self.assertEqual(response.status_code, 200)
