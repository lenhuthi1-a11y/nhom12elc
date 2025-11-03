from django.contrib import admin
from django.urls import path, include  # <-- THÊM 'include'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app.urls')),
]