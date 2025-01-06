from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase


class ViewTests(APITestCase):
    def test_example_view(self):
        url = reverse("example-view-name")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
