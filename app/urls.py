# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.setup_session, name='setup_session'),
    path('session/<int:session_id>/', views.run_session, name='run_session'),

    # === THÊM DÒNG NÀY ===
    path('session/<int:session_id>/export_csv/', views.export_session_csv, name='export_session_csv'),

    path('api/recognize', views.recognize_api, name='recognize_api'),
]