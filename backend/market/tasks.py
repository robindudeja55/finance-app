import os
import datetime as dt
from decimal import Decimal, InvalidOperation
import pandas as pd
import yfinance as yf
import joblib

from celery import shared_task
from django.db import transaction, DatabaseError
from django.db.utils import ProgrammingError, OperationalError
from .models import Instrument, PriceOHLCV, FeatureDaily

# Optional import of NewsArticle
try:
    from .models import NewsArticle
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def safe_decimal(val):
    try:
        if val is None or pd.isna(val):
            return None
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return None


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten MultiIndex columns like ('AAPL','Open') -> 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

    # Case-insensitive lookup, allow 'Adj Close' if 'Close' missing
    lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    o = pick("Open")
    h = pick("High")
    l = pick("Low")
    c = pick("Close", "Adj Close", "Adj_Close", "AdjClose")
    v = pick("Volume")

    missing = [name for name, colname in (("Open", o), ("High", h), ("Low", l), ("Close/AdjClose", c), ("Volume", v)) if colname is None]
    if missing:
        raise KeyError(f"Missing OHLCV columns: {missing}; got {list(df.columns)}")

    df = df.rename(columns={o: "Open", h: "High", l: "Low", c: "Close", v: "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


@shared_task
def fetch_prices(symbol: str = "AAPL", days: int = 400):
    end = dt.date.today()
    start = end - dt.timedelta(days=days)

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days}d", interval="1d", auto_adjust=True)

    if df is None or df.empty:
        return f"No data for {symbol}"

    df = normalize_ohlc(df)

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df["Volume"] = df["Volume"].fillna(0)

    inst, _ = Instrument.objects.get_or_create(symbol=symbol)

    rows = 0
    with transaction.atomic():
        for idx, row in df.iterrows():
            o = safe_decimal(row["Open"])
            h = safe_decimal(row["High"])
            l = safe_decimal(row["Low"])
            c = safe_decimal(row["Close"])
            if None in (o, h, l, c):
                continue
            vol = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
            PriceOHLCV.objects.update_or_create(
                instrument=inst,
                date=idx.date() if hasattr(idx, "date") else pd.to_datetime(idx).date(),
                defaults=dict(open=o, high=h, low=l, close=c, volume=vol),
            )
            rows += 1

    return f"{symbol}: upserted {rows} rows ({start} → {end})"


@shared_task
def build_features(symbol: str = "AAPL", days: int = 730):
    end = dt.date.today()
    start = end - dt.timedelta(days=days)

    inst, _ = Instrument.objects.get_or_create(symbol=symbol)

    q = (PriceOHLCV.objects
         .filter(instrument=inst, date__gte=start, date__lte=end)
         .order_by("date")
         .values("date", "close"))
    df = pd.DataFrame(list(q))
    if df.empty:
        return f"No prices for {symbol}. Run fetch_prices first."

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    df["close"] = df["close"].astype(float)

    # Price features
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    df["ma5_rel"] = (df["close"] / ma5) - 1.0
    df["ma20_rel"] = (df["close"] / ma20) - 1.0
    df["vol_5"] = df["ret_1d"].rolling(5).std()

    # Sentiment (optional)
    df["sentiment_day"] = 0.0
    if HAS_NEWS:
        try:
            nq = (NewsArticle.objects
                  .filter(instrument=inst,
                          published_at__date__gte=start,
                          published_at__date__lte=end)
                  .values("published_at", "sentiment"))
            nd = pd.DataFrame(list(nq))
            if not nd.empty:
                nd["date"] = pd.to_datetime(nd["published_at"]).dt.normalize()
                daily = nd.groupby("date")["sentiment"].mean()
                df = df.join(daily.rename("sentiment_day"), how="left")
                df["sentiment_day"] = df["sentiment_day"].fillna(0.0)
        except DatabaseError as e:
            print(f"build_features: skipping news join ({type(e).__name__}: {e})")

    # Label: 1 if next-day close is higher
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    keep = ["ret_1d", "ret_5d", "ma5_rel", "ma20_rel", "vol_5", "sentiment_day", "target_up"]
    df = df[keep].dropna().reset_index()

    inserted = 0
    try:
        with transaction.atomic():
            for row in df.to_dict(orient="records"):
                FeatureDaily.objects.update_or_create(
                    instrument=inst, date=row["date"].date(),
                    defaults=dict(
                        ret_1d=float(row["ret_1d"]),
                        ret_5d=float(row["ret_5d"]),
                        ma5_rel=float(row["ma5_rel"]),
                        ma20_rel=float(row["ma20_rel"]),
                        vol_5=float(row["vol_5"]),
                        sentiment_day=float(row["sentiment_day"]),
                        target_up=bool(row["target_up"]),
                    ),
                )
                inserted += 1
    except (ProgrammingError, OperationalError) as e:
        return f"DB error writing FeatureDaily (did you run migrations?): {e}"

    return f"{symbol}: upserted {inserted} feature rows"


@shared_task
def train_model(symbol: str = "AAPL", test_days: int = 60):
    inst, _ = Instrument.objects.get_or_create(symbol=symbol)
    rows = list(
        FeatureDaily.objects
        .filter(instrument=inst)
        .order_by("date")
        .values("date","ret_1d","ret_5d","ma5_rel","ma20_rel","vol_5","sentiment_day","target_up")
    )
    if not rows:
        return f"No features for {symbol}. Run build_features first."

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    if len(df) < test_days + 30:
        return f"Not enough rows ({len(df)}) for split."

    feats = ["ret_1d","ret_5d","ma5_rel","ma20_rel","vol_5","sentiment_day"]
    X = df[feats].astype(float)
    y = df["target_up"].astype(int)

    split = len(df) - test_days
    Xtr, ytr = X.iloc[:split], y.iloc[:split]
    Xte, yte = X.iloc[split:], y.iloc[split:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=1))
    ])
    pipe.fit(Xtr, ytr)

    acc = accuracy_score(yte, pipe.predict(Xte))
    try:
        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:,1])
    except Exception:
        auc = None

    model_dir = os.environ.get("MODEL_DIR", "artifacts")
    out_dir = os.path.join(model_dir, symbol)
    os.makedirs(out_dir, exist_ok=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = os.path.join(out_dir, f"logreg_{ts}.pkl")

    joblib.dump({
        "model": pipe,
        "features": feats,
        "trained_at": ts
    }, path)

    return (f"trained {symbol}: rows={len(df)}, test_days={test_days}, "
            f"acc={acc:.3f}, auc={(auc if auc is not None else 'n/a')}, saved={path}")
