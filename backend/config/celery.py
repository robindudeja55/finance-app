from celery import Celery
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('config')

app.config_from_object('django.conf:settings', namespace='CELERY')

# This imports tasks.py files in all registered Django apps so Celery can find tasks
app.autodiscover_tasks()






