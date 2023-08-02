from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("session", views.session, name="session"),
    path("config", views.config, name="config"),
    path("chat-process", views.chat_process, name="config"),
]