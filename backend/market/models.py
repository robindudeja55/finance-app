from django.db import models

class Instrument(models.Model):
    symbol = models.CharField(max_length=12, unique=True)
    name = models.CharField(max_length=128, blank=True)

    def __str__(self):
        return self.symbol



class PriceOHLCV(models.Model):
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, related_name="prices")
    date = models.DateField()
    open = models.DecimalField(max_digits=14, decimal_places=6)
    high = models.DecimalField(max_digits=14, decimal_places=6)
    low = models.DecimalField(max_digits=14, decimal_places=6)
    close = models.DecimalField(max_digits=14, decimal_places=6)
    volume = models.BigIntegerField()

    class Meta:
        unique_together = (("instrument", "date"),)
        indexes = [models.Index(fields=["instrument", "date"])]
        ordering = ["-date"]

class NewsArticle(models.Model):
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, related_name="news")
    title = models.CharField(max_length=512)
    summary = models.TextField(blank=True)
    url = models.URLField(unique=True)
    source = models.CharField(max_length=128, blank=True)
    published_at = models.DateTimeField()
    sentiment = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["instrument", "published_at"]),
            models.Index(fields=["published_at"]),
        ]
        ordering = ["-published_at"]

######################################################

class FeatureDaily(models.Model):
    instrument = models.ForeignKey(Instrument, on_delete=models.CASCADE, related_name="features")
    date = models.DateField()

    # price features
    ret_1d = models.FloatField(null=True)
    ret_5d = models.FloatField(null=True)
    ma5_rel = models.FloatField(null=True)    # close / MA5 - 1
    ma20_rel = models.FloatField(null=True)   # close / MA20 - 1
    vol_5 = models.FloatField(null=True)      # std of 1d return over 5d

    # news feature (optional)
    sentiment_day = models.FloatField(default=0.0)

    # label for training
    target_up = models.BooleanField()

    class Meta:
        unique_together = (("instrument", "date"),)
        indexes = [models.Index(fields=["instrument", "date"])]
        ordering = ["-date"]




class Prediction(models.Model):
    class Signal(models.TextChoices):
        UP = "UP"
        DOWN = "DOWN"
        HOLD = "HOLD"

    instrument = models.ForeignKey(
        'Instrument',
        on_delete=models.CASCADE,
        related_name="predictions"
    )
    date = models.DateField()
    prob_up = models.FloatField()
    signal = models.CharField(
        max_length=8,
        choices=Signal.choices,
        default=Signal.HOLD
    )
    model_name = models.CharField(max_length=64, default="logreg")
    trained_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (("instrument", "date", "model_name"),)
        indexes = [models.Index(fields=["instrument", "date"])]
        ordering = ["-date"]

    def __str__(self):
        return f"{self.instrument} — {self.date} — {self.signal}"

