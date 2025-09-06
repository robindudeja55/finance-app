import os
import glob
import datetime as dt
from decimal import Decimal, InvalidOperation
import logging

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from django.utils import timezone
from django.db import transaction, DatabaseError
from django.db.utils import ProgrammingError, OperationalError

from celery import shared_task, chain

from .models import (
    Instrument,
    PriceOHLCV,
    FeatureDaily,
    Prediction
)

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

logger = logging.getLogger("market.tasks")


# ---------------------------
# Helpers
# ---------------------------





def safe_decimal(val):
    try:
        if val is None or pd.isna(val):
            return None
        return Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return None


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

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

    missing = [name for name, colname in (("Open", o), ("High", h), ("Low", l),
                                          ("Close/AdjClose", c), ("Volume", v)) if colname is None]
    if missing:
        raise KeyError(f"Missing OHLCV columns: {missing}; got {list(df.columns)}")

    df = df.rename(columns={o: "Open", h: "High", l: "Low", c: "Close", v: "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]

# ---------------------------
# Task Scheduling Helpers
# ---------------------------  

def get_symbols():
    s = os.environ.get("SYMBOLS", "AAPL")
    return [x.strip().upper() for x in s.split(",") if x.strip()]

@shared_task
def scheduled_fetch_all(days: int = 365):
    for sym in get_symbols():
        fetch_prices.delay(sym, days)
    return f"enqueued fetch for {get_symbols()}"

@shared_task
def scheduled_build_features(days: int = 730):
    for sym in get_symbols():
        build_features.delay(sym, days)
    return f"enqueued features for {get_symbols()}"

@shared_task
def scheduled_train_all(test_days: int = 60):
    for sym in get_symbols():
        train_model.delay(sym, test_days)
    return f"enqueued train for {get_symbols()}"

@shared_task
def scheduled_predict_all():
    for sym in get_symbols():
        predict_next_day.delay(sym)
    return f"enqueued predict for {get_symbols()}"

##
############################
# ---------------------------
# Task 1: Fetch Prices
# ---------------------------
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


# ---------------------------
# Task 2: Build Features
# ---------------------------
@shared_task
def build_features(symbol: str = "AAPL", days: int = 730):
    logger.info("build_features.start symbol=%s days=%s", symbol, days)
    end = dt.date.today()
    start = end - dt.timedelta(days=max(1, int(days)))

    inst, _ = Instrument.objects.get_or_create(symbol=symbol)

    q = (PriceOHLCV.objects
         .filter(instrument=inst, date__gte=start, date__lte=end)
         .order_by("date")
         .values("date", "close"))
    df = pd.DataFrame(list(q))

    if df.empty:
        msg = f"No prices for {symbol}. Run fetch_prices first."
        logger.warning("build_features.empty symbol=%s", symbol)
        return msg

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()
    df["close"] = df["close"].astype(float)

    if len(df) < 25:
        msg = f"Not enough rows for features ({len(df)}) for {symbol}."
        logger.warning("build_features.too_few_rows symbol=%s rows=%s", symbol, len(df))
        return msg

    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    ma5 = df["close"].rolling(window=5, min_periods=5).mean()
    ma20 = df["close"].rolling(window=20, min_periods=20).mean()
    df["ma5_rel"] = (df["close"] / ma5) - 1.0
    df["ma20_rel"] = (df["close"] / ma20) - 1.0
    df["vol_5"] = df["ret_1d"].rolling(window=5, min_periods=5).std()

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
            logger.info("build_features.news_skipped symbol=%s err=%s", symbol, e)

    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)

    keep = ["ret_1d", "ret_5d", "ma5_rel", "ma20_rel", "vol_5", "sentiment_day", "target_up"]
    out = df[keep].dropna().reset_index()

    if out.empty:
        msg = f"No valid feature rows for {symbol} after rolling windows."
        logger.warning("build_features.no_valid_rows symbol=%s", symbol)
        return msg

    inserted = 0
    try:
        with transaction.atomic():
            for row in out.to_dict(orient="records"):
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
        logger.error("build_features.db_error symbol=%s err=%s", symbol, e)
        return f"DB error writing FeatureDaily: {e}"

    logger.info("build_features.done symbol=%s rows=%s", symbol, inserted)
    return f"{symbol}: upserted {inserted} feature rows"


# ---------------------------
# Task 3: Train Model
# ---------------------------
@shared_task
def train_model(symbol: str = "AAPL", test_days: int = 60):
    logger.info("train_model.start symbol=%s test_days=%s", symbol, test_days)

    inst, _ = Instrument.objects.get_or_create(symbol=symbol)
    rows = list(
        FeatureDaily.objects
        .filter(instrument=inst)
        .order_by("date")
        .values("date", "ret_1d", "ret_5d", "ma5_rel", "ma20_rel", "vol_5", "sentiment_day", "target_up")
    )

    if not rows:
        msg = f"No features for {symbol}. Run build_features first."
        logger.warning("train_model.no_features symbol=%s", symbol)
        return msg

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    if len(df) < test_days + 30:
        msg = f"Not enough rows ({len(df)}) for split."
        logger.warning("train_model.too_few_rows symbol=%s rows=%s", symbol, len(df))
        return msg

    feats = ["ret_1d", "ret_5d", "ma5_rel", "ma20_rel", "vol_5", "sentiment_day"]
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
        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    except Exception:
        auc = None

    model_dir = os.environ.get("MODEL_DIR", "artifacts")
    out_dir = os.path.join(model_dir, symbol)
    os.makedirs(out_dir, exist_ok=True)

    ts = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = os.path.join(out_dir, f"logreg_{ts}.pkl")
    joblib.dump({"model": pipe, "features": feats, "trained_at": ts}, path)

    logger.info(
        "train_model.done symbol=%s acc=%.3f auc=%s path=%s",
        symbol, acc, f"{auc:.3f}" if auc is not None else "n/a", path
    )
    return f"trained {symbol}: rows={len(df)}, test_days={test_days}, acc={acc:.3f}, auc={(auc if auc is not None else 'n/a')}, saved={path}"


# ---------------------------
# Helper for latest model
# ---------------------------
def _latest_model_path(symbol: str, model_dir: str | None = None) -> str | None:
    model_dir = model_dir or os.environ.get("MODEL_DIR", "artifacts")
    folder = os.path.join(model_dir, symbol)
    files = sorted(glob.glob(os.path.join(folder, "*.pkl")), reverse=True)
    return files[0] if files else None


# ---------------------------
# Task 4: Predict Next Day
# ---------------------------
@shared_task
def predict_next_day(symbol: str = "AAPL", thr_up: float = 0.55, thr_down: float = 0.45):
    logger.info("predict.start symbol=%s", symbol)

    inst, _ = Instrument.objects.get_or_create(symbol=symbol)
    latest = FeatureDaily.objects.filter(instrument=inst).order_by("-date").first()
    if not latest:
        msg = f"No features for {symbol}. Build features first."
        logger.warning("predict.no_features symbol=%s", symbol)
        return msg

    path = _latest_model_path(symbol)
    if not path:
        msg = f"No model artifact for {symbol}. Train model first."
        logger.warning("predict.no_model symbol=%s", symbol)
        return msg

    try:
        bundle = joblib.load(path)
        model = bundle["model"]
        features = bundle.get("features", ["ret_1d", "ret_5d", "ma5_rel", "ma20_rel", "vol_5", "sentiment_day"])
    except Exception as e:
        logger.error("predict.load_failed symbol=%s err=%s", symbol, e)
        return f"Failed to load model: {e}"

    values = []
    missing = []
    for f in features:
        v = getattr(latest, f, None)
        if v is None:
            missing.append(f)
        else:
            values.append(float(v))
    if missing:
        msg = f"Latest feature row missing fields {missing} for {symbol}"
        logger.warning("predict.missing_fields symbol=%s missing=%s", symbol, missing)
        return msg

    try:
        if hasattr(model, "predict_proba"):
            prob_up = float(model.predict_proba([values])[0][1])
        else:
            y = model.decision_function([values])
            prob_up = float(1 / (1 + np.exp(-y)))
    except Exception as e:
        logger.error("predict.infer_failed symbol=%s err=%s", symbol, e)
        return f"Inference failed: {e}"

    if prob_up >= thr_up:
        signal = "UP"
    elif prob_up <= thr_down:
        signal = "DOWN"
    else:
        signal = "HOLD"

    ta = bundle.get("trained_at")
    try:
        from datetime import datetime, timezone as dtz
        trained_at = datetime.strptime(ta, "%Y%m%d%H%M%S").replace(tzinfo=dtz.utc) if ta else timezone.now()
    except Exception:
        trained_at = timezone.now()

    Prediction.objects.update_or_create(
        instrument=inst,
        date=latest.date,
        model_name="logreg",
        defaults=dict(prob_up=prob_up, signal=signal, trained_at=trained_at),
    )

    logger.info("predict.done symbol=%s date=%s prob_up=%.3f signal=%s", symbol, latest.date, prob_up, signal)
    return f"{symbol}: date={latest.date} prob_up={prob_up:.3f} signal={signal} model={os.path.basename(path)}"


# ---------------------------
# Task 5: Run Full Pipeline
# ---------------------------
@shared_task
def run_full_pipeline(symbol: str = "AAPL"):
    """
    Runs the entire pipeline for a given symbol:
    fetch_prices → build_features → train_model → predict_next_day
    """
    workflow = chain(
        fetch_prices.s(symbol),
        build_features.s(symbol),
        train_model.s(symbol),
        predict_next_day.s(symbol)
    )
    result = workflow.apply_async()
    return f"Pipeline started for {symbol}, task_id={result.id}"
