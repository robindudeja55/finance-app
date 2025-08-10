from django.db import models
from django.utils import timezone


class ImportFile(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        PROCESSING = "PROCESSING", "Processing"
        DONE = "DONE", "Done"
        FAILED = "FAILED", "Failed"

    original_name = models.CharField(max_length=255)
    file = models.FileField(upload_to="imports/%Y/%m/%d/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    rows_total = models.IntegerField(default=0)
    rows_parsed = models.IntegerField(default=0)
    error = models.TextField(blank=True)

    def __str__(self):
        return f"{self.id} — {self.original_name} ({self.status})"


class Transaction(models.Model):
    import_file = models.ForeignKey(
        ImportFile,
        on_delete=models.CASCADE,
        related_name="transactions"
    )
    txn_date = models.DateField()
    description = models.CharField(max_length=255)
    merchant = models.CharField(max_length=255, blank=True)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    currency = models.CharField(max_length=3, default="USD")
    category = models.CharField(max_length=64, blank=True)
    account = models.CharField(max_length=64, blank=True)
    dedupe_hash = models.CharField(max_length=64, unique=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        indexes = [
            models.Index(fields=["txn_date"]),
            models.Index(fields=["merchant"]),
            models.Index(fields=["category"]),
        ]
        ordering = ["-txn_date", "-id"]

    def __str__(self):
        return f"{self.txn_date} {self.amount} {self.description[:30]}"
