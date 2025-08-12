from django.contrib import admin
from .models import Instrument, PriceOHLCV, FeatureDaily, Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("instrument", "date", "prob_up", "signal", "model_name", "trained_at")
    list_filter = ("instrument", "model_name", "signal")
    search_fields = ("instrument__symbol",)
    date_hierarchy = "date"
