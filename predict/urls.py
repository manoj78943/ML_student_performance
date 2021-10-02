from django.urls import path
from.import views
from django.contrib import admin

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.welcome, name="welcome"),
    path('result', views.result, name="result"),
]