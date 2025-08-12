from django.urls import path
from . import views

urlpatterns = [
    path("api/prediction", views.prediction_latest, name="prediction-latest"),
    path("api/price-series", views.price_series, name="price-series"),
]