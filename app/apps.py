from django.apps import AppConfig
#pip install -r requirements.txt

class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
