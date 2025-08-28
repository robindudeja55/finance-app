from datetime import date, timedelta
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .models import Instrument, PriceOHLCV, Prediction
from django.views.decorators.http import require_GET

@require_GET
def prediction_latest(request):
    symbol = request.GET.get("symbol", "AAPL").upper()
    try:
        inst = Instrument.objects.get(symbol=symbol)
    except Instrument.DoesNotExist:
        return JsonResponse({"error": f"unknown symbol {symbol}"}, status=404)

    pred = Prediction.objects.filter(instrument=inst).order_by("-date").first()
    if not pred:
        return JsonResponse({"symbol": symbol, "prediction": None, "message": "no prediction yet"}, status=200)

    return JsonResponse({
        "symbol": symbol,
        "date": pred.date.isoformat(),
        "prob_up": float(pred.prob_up),
        "signal": pred.signal,
        "model": pred.model_name,
        "trained_at": pred.trained_at.isoformat() if pred.trained_at else None,
    })

@require_GET
def price_series(request):
    symbol = request.GET.get("symbol", "AAPL").upper()
    days = int(request.GET.get("days", "60"))

    try:
        inst = Instrument.objects.get(symbol=symbol)
    except Instrument.DoesNotExist:
        return JsonResponse({"error": f"unknown symbol {symbol}"}, status=404)

    start = date.today() - timedelta(days=days + 10)
    qs = (PriceOHLCV.objects
          .filter(instrument=inst, date__gte=start)
          .order_by("date")
          .values("date", "close"))

    data = [{"date": row["date"].isoformat(), "close": float(row["close"])} for row in qs]
    return JsonResponse({"symbol": symbol, "series": data})


@require_GET
def symbols(request):
    syms = list(Instrument.objects.order_by("symbol").values_list("symbol", flat=True))
    if not syms:
        syms = [s.strip().upper() for s in os.environ.get("SYMBOLS", "AAPL").split(",") if s.strip()]
    return JsonResponse({"symbols": syms})

################################################