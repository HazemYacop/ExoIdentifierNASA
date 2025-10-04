from django.urls import path

from . import views

app_name = "main"

urlpatterns = [
    path("", views.home, name="home"),
    path("model/", views.model, name="model"),
    path("api/analyze/", views.analyze_image, name="analyze"),
]

