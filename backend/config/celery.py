from celery import Celery
from celery.schedules import crontab
import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

app = Celery('config')

app.config_from_object('django.conf:settings', namespace='CELERY')

# This imports tasks.py files in all registered Django apps so Celery can find tasks
app.autodiscover_tasks()

# Set timezone for beat schedules
app.conf.timezone = os.environ.get("TIME_ZONE", "UTC")

# Define scheduled periodic tasks
app.conf.beat_schedule = {
    "fetch-prices-hourly": {
        "task": "market.tasks.scheduled_fetch_all",
        "schedule": crontab(minute=5),  # every hour at :05
        "args": (365,),
    },
    "build-features-daily": {
        "task": "market.tasks.scheduled_build_features",
        "schedule": crontab(hour=21, minute=15),  # daily at 21:15
        "args": (730,),
    },
    "train-weekly": {
        "task": "market.tasks.scheduled_train_all",
        "schedule": crontab(hour=21, minute=30, day_of_week="sun"),  # Sundays 21:30
        "args": (60,),
    },
    "predict-daily": {
        "task": "market.tasks.scheduled_predict_all",
        "schedule": crontab(hour=21, minute=20),  # daily at 21:20
    },
}






