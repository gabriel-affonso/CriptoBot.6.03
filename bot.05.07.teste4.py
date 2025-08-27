#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pipeline_combined_bot.py

"""
A live trading bot that only enters long positions when both:
  1) The “math‐based” indicator strategy (EMA/MACD/rule‐based) fires a signal.
  2) The IA model (XGBoost‐stacked classifier) predicts a high probability of positive next‐minute return.
  
All indicators used (EMA9, EMA50, EMA200, MACD, RSI, etc.) are computed on 15m and 1h data exactly as in the training pipeline, so that the IA model sees the same features it was trained on.
If both “math” and “IA” agree on a long entry, the bot submits a market BUY to Binance.  
Position management uses partial exits at +0.8%, breakeven at +0.5%, full take‐profit at +2%, and a 0.5% stop‐loss.  
It tracks equity in USDC, adjusts risk dynamically based on drawdown/growth, and logs every trade to CSV + Telegram.
"""

import sys
print(sys.executable)  # show which Python interpreter is running
import os
import time
import json
import csv
import traceback
import threading
import logging
from decimal import Decimal, ROUND_DOWN, DecimalException
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, DMatrix, Booster
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import pearsonr

from binance.client import Client
from binance.exceptions import BinanceAPIException
import skops.io as sio

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────


API_KEY = ''
API_SECRET = ''
TELEGRAM_TOKEN = ''
TELEGRAM_CHAT_ID = ''

PAIRS = [
    'BTCUSDC', 'ETHUSDC', 'BNBUSDC', 'XRPUSDC', 'ADAUSDC',
    'SOLUSDC', 'DOGEUSDC', 'AVAXUSDC', 'DOTUSDC', 'TRXUSDC',
    'LINKUSDC', 'MATICUSDC', 'LTCUSDC', 'SHIBUSDC', 'UNIUSDC',
    'BCHUSDC', 'ICPUSDC', 'ETCUSDC', 'XLMUSDC', 'FILUSDC'
]
# Trading parameters (carry over from your backtest)
BASE_RISK       = 0.03    # 1.5% of equity at risk per trade
SL_PCT_LONG     = 0.005   # 0.5% stop‐loss below entry for long
TP_PCT          = 0.06     # 2% take‐profit
BE_PCT          = 0.002    # 0.5% breakeven trigger for long
DD_STOP_PCT     = 0.10     # 10% drawdown threshold to reduce risk
EMA_SHORT       = 9        # EMA(9) for 15m bars
EMA_MEDIUM      = 50       # EMA(50)
EMA_LONG        = 200      # EMA(200)
PARTIAL_PCT     = Decimal('0.003')   # +0.8% partial take‐profit
PARTIAL_SIZE    = Decimal('0.50')    # sell 50% at partial
MIN_USDC_TRADE  = 5        # absolute minimum USDC per trade
MIN_TRADE_BY_PAIR = {
    'BTCUSDT': 10, 'ETHUSDT': 5, 'ADAUSDT': 2,
    'SOLUSDT': 3, 'XRPUSDT': 2
    # others default to MIN_USDC_TRADE if not listed
}

INITIAL_CAPITAL = None
current_equity  = 0.0
equity_high     = 0.0

STATE_FILE = 'bot_positions_state.json'
LOG_FILE   = 'bot_trades_log.csv'

# ─── MODEL ARTIFACTS ────────────────────────────────────────────────────────────

# Make sure this directory holds your saved scaler, regressor, classifier, etc.
MODEL_DIR      = r"C:\Users\CES\Dropbox\Coisas\Coisas do PC\4\4.18_gpu_v5"
SCALER_PATH    = os.path.join(MODEL_DIR, 'scaler.joblib')
REG_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_reg.json')
CLF_MODEL_PATH = os.path.join(MODEL_DIR, 'stacked_clf_v5.pkl')
# SCALER_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'scaler_classifier.joblib')  # NEW: classifier scaler

# Random Forest artifacts
RF_ARTIFACTS_DIR = r"C:\Users\CES\Dropbox\Coisas\Coisas do PC\4\6.01"
RF_SKOPS_PATH    = os.path.join(RF_ARTIFACTS_DIR, 'rf_model.skops')
RF_FEATURES_PATH = os.path.join(RF_ARTIFACTS_DIR, 'rf_feature_columns.txt')

# Thresholds
CLF_THRESHOLD   = 0.69
IA_REG_THRESHOLD = 0.01

# Skip logging
SKIP_LOG_FILE = 'bot_skip_log.csv'

# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────

def setup_logging():
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt))
    lg.addHandler(ch)
    fh = logging.FileHandler(os.path.join(MODEL_DIR, "bot_process .log"), mode='w')
    fh.setFormatter(logging.Formatter(fmt))
    lg.addHandler(fh)

# Utility: log trades to CSV
def register_trade(tipo, symbol, qty, price, reason):
    header = ['timestamp','type','symbol','qty','price','reason']
    newrow = [
        datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        tipo, symbol, f"{qty:.6f}", f"{price:.4f}", reason
    ]
    write_header = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(newrow)
    logging.info(f"[LOG] {tipo} {symbol} qty={qty:.6f} @ {price:.4f} | Reason: {reason}")

# Utility: send a Telegram message
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.get(url, params={'chat_id': TELEGRAM_CHAT_ID, 'text': msg}, timeout=5)
    except Exception:
        pass

def log_skip(layer: str, symbol: str, info: str = ""):
    """Register skip events to CSV and Telegram."""
    msg = f"[SKIP][{layer}] {symbol} {info}".strip()
    logging.info(msg)
    send_telegram(msg)
    header = ['timestamp', 'layer', 'symbol', 'info']
    newrow = [datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), layer, symbol, info]
    write_header = not os.path.exists(SKIP_LOG_FILE)
    with open(SKIP_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(newrow)

# --- FEATURE LIST LOADING AND SANITY CHECK ---
FEATURE_COLUMNS_TXT = os.path.join(MODEL_DIR, 'feature_columns.txt')
print(f"[DEBUG] Verificando {FEATURE_COLUMNS_TXT} ...")
if os.path.exists(FEATURE_COLUMNS_TXT):
    with open(FEATURE_COLUMNS_TXT) as f:
        feature_columns_txt = [line.strip() for line in f if line.strip()]
    print("[DEBUG] feature_columns.txt carregado:", feature_columns_txt)
    logging.info(f"[FEATURES] feature_columns.txt loaded: {feature_columns_txt}")
else:
    feature_columns_txt = None
    print(f"[DEBUG] feature_columns.txt NÃO encontrado em {FEATURE_COLUMNS_TXT}")
    logging.warning(f"[FEATURES] feature_columns.txt not found at {FEATURE_COLUMNS_TXT}")

def validate_model_features(model, features):
    if hasattr(model, 'feature_names_in_'):
        missing = set(model.feature_names_in_) - set(features)
        extra = set(features) - set(model.feature_names_in_)
        if missing:
            raise ValueError(f"Features ausentes no DataFrame: {missing}")
        if extra:
            logging.warning(f"[FEATURES] Features extras no DataFrame: {extra}")
    else:
        logging.warning("[FEATURES] Model does not have feature_names_in_ attribute.")

# ─── STATE MANAGEMENT ──────────────────────────────────────────────────────────

positions = {}       # positions[symbol] → {fields…} or None if no position
position_times = {}  # position_times[symbol] → datetime when entered

def load_state():
    """Load persisted positions from disk (adds 'partial_taken', 'be_moved', etc.)."""
    global positions, position_times
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        # Reconstruct positions + times
        saved_pos = data.get('positions', {})
        saved_times= data.get('times', {})
        for s in PAIRS:
            pos = saved_pos.get(s, None)
            if pos:
                # Ensure all required keys exist
                pos.setdefault('side', 'long')
                pos.setdefault('entry_price', pos.get('entry_price', 0.0))
                pos.setdefault('quantity', pos.get('quantity', 0.0))
                pos.setdefault('sl', pos.get('sl', 0.0))
                pos.setdefault('tp', pos.get('tp', 0.0))
                pos.setdefault('be_trigger', pos.get('be_trigger', 0.0))
                pos.setdefault('be_moved', False)
                pos.setdefault('risk', pos.get('risk', None))
                pos.setdefault('partial_taken', False)
                positions[s] = pos
            else:
                positions[s] = None
            t = saved_times.get(s, None)
            if t:
                position_times[s] = datetime.fromisoformat(t)
            else:
                position_times[s] = None
    else:
        # Initialize empty
        for s in PAIRS:
            positions[s] = None
            position_times[s] = None

def save_state():
    """Atomically write out the current positions + position_times to disk."""
    data = {
        'positions': {s: positions[s] for s in PAIRS if positions[s]},
        'times': {s: position_times[s].isoformat() for s in PAIRS if position_times[s]}
    }
    tmp = STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    if os.path.exists(tmp):
        os.replace(tmp, STATE_FILE)

# ─── BINANCE SETUP ─────────────────────────────────────────────────────────────

client = Client(API_KEY, API_SECRET)
symbol_rules_cache = {}

def get_symbol_rules(symbol):
    """
    Returns a dict with:
      - step_size (Decimal)
      - min_qty    (Decimal)
      - min_notional (Decimal)
    from Binance’s filters. Caches results.
    """
    if symbol in symbol_rules_cache:
        return symbol_rules_cache[symbol]
    info = client.get_symbol_info(symbol)
    lot = next(f for f in info['filters'] if f['filterType']=='LOT_SIZE')
    noti= next(f for f in info['filters'] if f['filterType']=='NOTIONAL')
    rules = {
        'step_size': Decimal(lot['stepSize']),
        'min_qty':    Decimal(lot['minQty']),
        'min_notional': Decimal(noti['minNotional'])
    }
    symbol_rules_cache[symbol] = rules
    return rules

# ─── INDICATOR CALCULATION ────────────────────────────────────────────────────

def fetch_klines_df(symbol: str, interval: str, limit: int):
    """
    Fetches Binance KLINES and returns a DataFrame with:
      ['open_time','open','high','low','close','volume','close_time', …].  
    All numeric columns are cast to float.  
    Last (incomplete) bar is dropped if needed.
    Adds logging to help diagnose candle recency issues.
    Supports batching for >1000 1m candles.
    """
    import time
    import pandas as pd
    from datetime import datetime, timezone
    # --- Batch fetch for 1m candles if limit > 1000 ---
    if interval == Client.KLINE_INTERVAL_1MINUTE and limit > 1000:
        all_klines = []
        batch_limit = 1000
        end_time = None
        remaining = limit
        while remaining > 0:
            fetch_limit = min(batch_limit, remaining)
            params = dict(symbol=symbol, interval=interval, limit=fetch_limit)
            if end_time:
                params['endTime'] = end_time
            klines = client.get_klines(**params)
            if not klines:
                break
            all_klines = klines[:-1] + all_klines if end_time else klines + all_klines
            # Prepare for next batch
            end_time = klines[0][0] - 1  # fetch older candles next
            remaining -= len(klines)
            if len(klines) < fetch_limit:
                break  # no more data
        klines = all_klines[-limit:]  # ensure we only return the most recent 'limit' candles
    else:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_vol','trades','taker_base_vol','taker_quote_vol','ignore'
    ])
    # Convert to proper dtypes
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    # Drop last if bar still open
    if not df.empty:
        last_ct = int(df.iloc[-1]['close_time'])
        if last_ct > int(time.time()*1000):
            df = df.iloc[:-1]
    # Convert open_time to pd.Timestamp (force UTC)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('open_time', inplace=True)
    # --- LOGGING: Check recency ---
    if not df.empty:
        last_candle_time = df.index[-1]
        # Compute last candle close time
        if interval == Client.KLINE_INTERVAL_15MINUTE:
            last_candle_close = last_candle_time + pd.Timedelta('15min')
            max_delay = pd.Timedelta('30min')
        elif interval == Client.KLINE_INTERVAL_1HOUR:
            last_candle_close = last_candle_time + pd.Timedelta('1h')
            max_delay = pd.Timedelta('75min')
        else:
            # Default: treat as 15m
            last_candle_close = last_candle_time + pd.Timedelta('15min')
            max_delay = pd.Timedelta('30min')
        now_utc = datetime.now(timezone.utc)
        logging.info(f"[{symbol}] System UTC: {now_utc}, Last {interval} candle: {last_candle_time} (closes {last_candle_close})")
        if (now_utc - last_candle_close) > max_delay:
            logging.warning(f"[{symbol}] Último candle é antigo: {last_candle_time} fecha {last_candle_close} (agora: {now_utc})")
            return pd.DataFrame()
    else:
        logging.warning(f"[{symbol}] DataFrame de candles vazio para {interval}.")
    return df

def compute_15m_indicators(df15: pd.DataFrame):
    """
    Given a 15m dataframe with columns ['open','high','low','close','volume'],
    computes:
      - ema9, ema50, ema200 on 15m close
      - vol_sma = sma(volume,20)
    Returns the augmented DataFrame with those columns.
    """
    df15['ema9']   = df15['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df15['ema50']  = df15['close'].ewm(span=EMA_MEDIUM, adjust=False).mean()
    df15['ema200'] = df15['close'].ewm(span=EMA_LONG, adjust=False).mean()
    df15['vol_sma']= df15['volume'].rolling(20).mean()
    return df15

def compute_1h_macd(df1h: pd.DataFrame):
    """
    Given a 1h DataFrame, computes MACD = EMA12-EMA26,
    MACD_SIGNAL = EMA(MACD,9). Returns the two Series.
    """
    close_1h = df1h['close']
    ema12 = close_1h.ewm(span=12, adjust=False).mean()
    ema26 = close_1h.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    return macd, macd_signal

# ─── LOAD PRETRAINED IA MODELS ─────────────────────────────────────────────────

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
setup_logging()
logging.info("Loading pretrained IA models (scaler, regressor, classifier)...")

# 1) Load scaler
scaler: StandardScaler = joblib.load(SCALER_PATH)

# Dynamically set feature_columns to match scaler
if hasattr(scaler, 'feature_names_in_'):
    feature_columns = list(scaler.feature_names_in_)
else:
    # fallback: use n_features_in_ and warn
    feature_columns = [f'f{i}' for i in range(scaler.n_features_in_)]
    logging.warning(f"[SCALER] feature_names_in_ not found, using generic names: {feature_columns}")

# Try to load a separate scaler for the classifier (68 features)
SCALER_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'scaler_classifier.joblib')
try:
    scaler_classifier: StandardScaler = joblib.load(SCALER_CLASSIFIER_PATH)
    logging.info(f"[SCALER_CLASSIFIER] Loaded classifier scaler from {SCALER_CLASSIFIER_PATH}")
except Exception as e:
    scaler_classifier = scaler  # fallback to regressor scaler
    logging.warning(f"[SCALER_CLASSIFIER] Could not load classifier scaler: {e}. Using regressor scaler as fallback (may cause feature mismatch errors!)")

# 2) Load XGBoost regression model (to predict next‐minute return)
#    We will only use direction_accuracy on reg if we ever want to log it; we rely on classifier probabilities.
regressor = DMatrix(np.zeros((1,1)))  # placeholder
xgb_reg = None
try:
    # Try to load as Booster directly
    xgb_reg = Booster()
    xgb_reg.load_model(REG_MODEL_PATH)
except Exception:
    logging.error("[MODEL LOAD ERROR] Could not load XGBoost regressor model.")

# 3) Load stacked classifier + calibration
stacked_clf: CalibratedClassifierCV = joblib.load(CLF_MODEL_PATH)

logging.info("IA models loaded successfully.")

# --- Random Forest model (SKOPS) ---
rf_model = None
rf_feature_columns: list[str] = []
try:
    if os.path.exists(RF_SKOPS_PATH):
        rf_model = sio.load(RF_SKOPS_PATH)
        logging.info(f"[RF] Loaded model from {RF_SKOPS_PATH}")
    else:
        logging.warning(f"[RF] RF model not found at {RF_SKOPS_PATH}")
except Exception as e:
    rf_model = None
    logging.exception(f"[RF] Failed to load model: {e}")

try:
    if os.path.exists(RF_FEATURES_PATH):
        with open(RF_FEATURES_PATH) as f:
            rf_feature_columns = [line.strip() for line in f if line.strip()]
        logging.info(f"[RF] Loaded feature columns ({len(rf_feature_columns)})")
    else:
        logging.warning(f"[RF] Feature list not found at {RF_FEATURES_PATH}")
except Exception as e:
    rf_feature_columns = []
    logging.exception(f"[RF] Could not load feature list: {e}")

RF_FEATURE_SAVE_DIR = os.path.join(RF_ARTIFACTS_DIR, 'runtime_rf_features')
RF_SAVE_FEATURES = False

def build_rf_features_15m(symbol: str) -> pd.DataFrame:
    """Build RF features using 1m candles resampled to 15m."""
    try:
        df1m = fetch_klines_df(symbol, Client.KLINE_INTERVAL_1MINUTE, limit=1000)
        if df1m.shape[0] < 20:
            logging.warning(f"[RF] {symbol} insufficient 1m data")
            return pd.DataFrame()
        df1m = df1m[['open','high','low','close','volume']].sort_index()
        df15 = df1m.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        if df15.empty:
            return pd.DataFrame()

        df15['return'] = df15['close'].pct_change()
        df15['return_5min'] = df1m['close'].pct_change(5).resample('15min').last()
        df15['roll_std_5min'] = df1m['close'].pct_change().rolling(5).std().resample('15min').last()
        df15['symbol_id'] = LabelEncoder().fit_transform(pd.Series([symbol]*len(df15)))
        df15['symbol_code'] = symbol
        df15['open_time_unix'] = df15.index.view(np.int64)//10**9

        feat = df15.iloc[[-1]].copy()
        if rf_feature_columns:
            missing = [c for c in rf_feature_columns if c not in feat.columns]
            if missing:
                logging.warning(f"[RF] {symbol} missing columns: {missing}")
                return pd.DataFrame()
            feat = feat.reindex(columns=rf_feature_columns)

        if feat.isnull().any().any():
            nan_cols = feat.columns[feat.isnull().any()].tolist()
            logging.warning(f"[RF] {symbol} NaN columns: {nan_cols}")
            return pd.DataFrame()

        if RF_SAVE_FEATURES:
            os.makedirs(RF_FEATURE_SAVE_DIR, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            feat.to_csv(os.path.join(RF_FEATURE_SAVE_DIR, f"{symbol}_{ts}.csv"), index=False)

        return feat
    except Exception as e:
        logging.exception(f"[RF] {symbol} feature build error: {e}")
        return pd.DataFrame()

def rf_model_validation(symbol: str) -> tuple[bool, int, np.ndarray]:
    """Validate entry using RF model. Approves only if class 2 predicted."""
    if rf_model is None:
        return (False, None, None)
    feat = build_rf_features_15m(symbol)
    if feat.empty:
        return (False, None, None)
    try:
        X = feat.to_numpy(dtype=np.float32)
        proba = rf_model.predict_proba(X)[0]
        pred_label = int(rf_model.predict(X)[0])
        logging.info(f"[RF] {symbol} pred_label={pred_label}, proba={proba.tolist()}")
        return (pred_label == 2, pred_label, proba)
    except Exception as e:
        logging.exception(f"[RF] {symbol} prediction error: {e}")
        return (False, None, None)

def combined_entry_check(symbol: str) -> bool:
    """Sequential entry gate with math, IA reg, classifier and RF."""
    math_ok = indicator_signal_long(symbol)
    if not math_ok:
        log_skip('MATH', symbol)
        return False
    ia_ok = ia_model_signal_long(symbol)
    if not ia_ok:
        log_skip('IA_REG', symbol)
        return False
    clf_ok = ia_model_classifier_validation(symbol, threshold=CLF_THRESHOLD)
    if not clf_ok:
        log_skip('CLF', symbol)
        return False
    rf_ok, rf_label, rf_proba = rf_model_validation(symbol)
    if not rf_ok or rf_label != 2:
        logging.info(f"[ENTRY] RF blocked: label={rf_label}")
        log_skip(f'RF label={rf_label}', symbol)
        return False
    logging.info(f"[ENTRY APPROVED] MATH+IA_REG+CLF+RF(2) {symbol}")
    send_telegram(f"[ENTRY APPROVED] MATH+IA_REG+CLF+RF(2) {symbol}")
    return True

# ─── ENTRY SIGNAL FUNCTION ─────────────────────────────────────────────────────

def indicator_signal_long(symbol: str) -> bool:
    """
    Returns True if at least three of the following five conditions fire a LONG
    signal:
      - 15m EMA50 > EMA200 AND 15m MACD > MACD_SIGNAL
      - 1h bar is bullish (close > open)
      - recent 3 bars of EMA50 on 15m are monotonic increasing
      - 15m volume >= 15m vol_sma
      - price > EMA9 > EMA50 on the latest 15m bar
    """
    # Fetch last 100 bars of 15m and 1h
    df15 = fetch_klines_df(symbol, Client.KLINE_INTERVAL_15MINUTE, limit=100)
    if df15.shape[0] < 50:
        return False
    df15 = compute_15m_indicators(df15)
    df1h = fetch_klines_df(symbol, Client.KLINE_INTERVAL_1HOUR, limit=50)
    if df1h.shape[0] < 30:
        return False
    macd, macd_signal = compute_1h_macd(df1h)

    # Latest 15m bar
    bar15 = df15.iloc[-1]
    price = bar15['close']

    # Condition 1: 15m EMA50 > EMA200 AND MACD > MACD_SIGNAL
    cond1 = (bar15['ema50'] > bar15['ema200']) and (macd.iloc[-1] > macd_signal.iloc[-1])
    # Condition 2: 1h bullish bar
    bar1h = df1h.iloc[-1]
    cond2 = bar1h['close'] > bar1h['open']
    # Condition 3: recent EMA50 monotonic increasing over last 3 bars
    if df15.shape[0] >= 4:
        ema50_recent = df15['ema50'].iloc[-4:-1]
        cond3 = ema50_recent.is_monotonic_increasing
    else:
        cond3 = False
    # Condition 4: 15m volume >= vol_sma
    cond4 = bar15['volume'] >= bar15['vol_sma']
    # Condition 5: price > EMA9 > EMA50
    cond5 = (price > bar15['ema9']) and (bar15['ema9'] > bar15['ema50'])

    # Score system: need at least 3 of 5 conditions to pass
    score = sum([cond1, cond2, cond3, cond4, cond5])
    return score >= 3

# ─── IA MODEL SIGNAL FUNCTION ──────────────────────────────────────────────────

prediction_log = []  # each entry: dict(symbol, bar_time, pred_return)

def record_prediction(symbol: str, bar_time: pd.Timestamp, pred_return: float):
    """Store each prediction so we can later compare it to the real move."""
    prediction_log.append({
        "symbol": symbol,
        "bar_time": bar_time,
        "pred_return": pred_return,
    })

def evaluate_predictions():
    """
    Look through prediction_log for any entries whose bar_time + 15m has elapsed,
    fetch that closing price, compute the actual return, and log MAE/DirAcc so far.
    """
    now = datetime.now(timezone.utc)  # Use timezone-aware UTC datetime
    window = timedelta(minutes=15)
    done = []
    errors = []
    correct_dir = 0

    for entry in prediction_log:
        target_time = entry["bar_time"] + window
        if now < target_time:
            continue

        # fetch exactly the two bars we need
        df15 = fetch_klines_df(entry["symbol"], Client.KLINE_INTERVAL_15MINUTE, limit=2)
        # make sure index is sorted
        df15 = df15.sort_index()

        # find the row at entry["bar_time"] and at target_time
        if entry["bar_time"] in df15.index and target_time in df15.index:
            close0 = df15.loc[entry["bar_time"],  "close"]
            close1 = df15.loc[target_time,         "close"]
            actual = (close1/close0) - 1.0
            errors.append(abs(actual - entry["pred_return"]))
            # direction‐accuracy
            if np.sign(actual) == np.sign(entry["pred_return"]):
                correct_dir += 1
            done.append(entry)

    # clean up evaluated
    for e in done:
        prediction_log.remove(e)

    if done:
        mae       = np.mean(errors)
        dir_acc   = correct_dir / len(done)
        msg = f"[EVAL] {len(done)} preds ⇒ MAE={mae:.4%}, DirAcc={dir_acc:.1%}"
        logging.info(msg)
        send_telegram(msg)

def ia_model_signal_long(symbol: str) -> bool:
    """
    Gera sinal de entrada LONG baseado na previsão de retorno do regressor XGBoost.
    Só retorna True se:
      1) Replique o pipeline de features de 1m → 15m exatamente como no treino.
      2) Prever retorno com xgb_reg e for ≥ 1% (0.01).
    """
    # 1) Busca 3000 candles de 1m e resample para 15m
    df1m = fetch_klines_df(symbol, Client.KLINE_INTERVAL_1MINUTE, limit=3000)
    print(f"[IA_MODEL] {symbol} df1m.shape={df1m.shape}")
    logging.info(f"[IA_MODEL] {symbol} df1m.shape={df1m.shape}")
    if df1m.shape[0] < 60:
        print(f"[IA_MODEL] {symbol} SKIP ENTRY: menos de 60 candles 1m disponíveis")
        logging.info(f"[IA_MODEL] {symbol} SKIP ENTRY: menos de 60 candles 1m disponíveis")
        return False
    df1m = df1m[['open','high','low','close','volume']].sort_index()
    df15 = df1m.resample('15min').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    }).dropna()
    print(f"[IA_MODEL] {symbol} df15.shape(before dropna)={df15.shape}")
    logging.info(f"[IA_MODEL] {symbol} df15.shape(before dropna)={df15.shape}")
    # 2) Recalcula todas as features usadas no treino
    df15['return']       = df15['close'].pct_change()
    df15['price_change'] = df15['close'] - df15['open']
    df15['volatility']   = df15['return'].rolling(14).std()
    df15['direction']    = np.sign(df15['return']).fillna(0)

    df15['sma_20']  = df15['close'].rolling(20).mean()
    df15['ema_50']  = df15['close'].ewm(span=50,  adjust=False).mean()
    df15['ema_9']   = df15['close'].ewm(span=9,   adjust=False).mean()  # ensure present
    df15['ema_200'] = df15['close'].ewm(span=200, adjust=False).mean()  # ensure present

    from ta.trend      import ADXIndicator
    from ta.momentum   import RSIIndicator, StochasticOscillator, TSIIndicator
    from ta.volume     import MFIIndicator
    from ta.volatility import BollingerBands

    df15['adx_14']   = ADXIndicator(df15['high'], df15['low'], df15['close'], window=14).adx()
    df15['rsi_14']   = RSIIndicator(df15['close'], window=14).rsi()
    sto              = StochasticOscillator(df15['high'], df15['low'], df15['close'], window=14, smooth_window=3)
    df15['sto_k']    = sto.stoch()
    df15['sto_d']    = sto.stoch_signal()
    df15['macd']     = df15['close'].ewm(span=12, adjust=False).mean() - df15['close'].ewm(span=26, adjust=False).mean()
    df15['macd_sig'] = df15['macd'].ewm(span=9, adjust=False).mean()
    df15['roc_10']   = df15['close'].pct_change(10)
    df15['tsi_25']   = TSIIndicator(df15['close'], window_slow=25, window_fast=13).tsi()

    hl = df15['high'] - df15['low']
    hc = (df15['high'] - df15['close'].shift()).abs()
    lc = (df15['low']  - df15['close'].shift()).abs()
    df15['atr_14']   = pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(14).mean()
    bb              = BollingerBands(df15['close'], window=20, window_dev=2)
    df15['bb_width']     = bb.bollinger_hband() - bb.bollinger_lband()
    df15['bb_percent_b'] = bb.bollinger_pband()

    df15['dc_width']     = df15['high'].rolling(20).max() - df15['low'].rolling(20).min()
    df15['vol_ma_20']    = df15['volume'].rolling(20).mean()
    df15['vol_ratio_20'] = df15['volume'] / df15['vol_ma_20']
    df15['obv']          = (np.sign(df15['close'].diff()) * df15['volume']).cumsum()
    df15['vpt']          = (df15['close'].pct_change() * df15['volume']).cumsum()

    for p in [5,10,20,60]:
        df15[f'ret_{p}']     = df15['close'].pct_change(p)
        df15[f'logret_{p}']  = np.log(df15['close']/df15['close'].shift(p))
        df15[f'roll_std_{p}']= df15['return'].rolling(p).std()
        df15[f'roll_skew_{p}']= df15['return'].rolling(p).skew()
    # Add roll_std_6 for compatibility with old scaler
    df15['roll_std_6'] = df15['return'].rolling(6).std()

    # Candle anatomy features
    df15['upper_shadow'] = df15['high'] - df15[['close','open']].max(axis=1)
    df15['lower_shadow'] = df15[['close','open']].min(axis=1) - df15['low']
    df15['body_size']    = (df15['close'] - df15['open']).abs()

    # Money Flow Index
    df15['mfi_14'] = MFIIndicator(df15['high'], df15['low'], df15['close'], df15['volume'], window=14).money_flow_index()

    # 5-min return alias (for return_5min)
    df15['return_5min'] = df15['close'].pct_change(5)

    df15['minute']            = df15.index.minute
    df15['hour']              = df15.index.hour
    df15['dayofweek']         = df15.index.dayofweek
    df15['dayofmonth']        = df15.index.day
    df15['month']             = df15.index.month
    df15['year']              = df15.index.year
    df15['day']               = df15.index.day
    df15['weekday']           = df15.index.weekday
    df15['is_month_end']      = df15.index.is_month_end.astype(int)
    df15['is_month_start']    = df15.index.is_month_start.astype(int)
    df15['is_quarter_end']    = df15.index.is_quarter_end.astype(int)
    df15['mins_since_daystart']= df15.index.hour*60 + df15.index.minute
    df15['hour_sin']          = np.sin(2*np.pi*df15['hour']/24)
    df15['hour_cos']          = np.cos(2*np.pi*df15['hour']/24)
    df15['dow_sin']           = np.sin(2*np.pi*df15['dayofweek']/7)
    df15['dow_cos']           = np.cos(2*np.pi*df15['dayofweek']/7)

    df15['symbol_id'] = LabelEncoder().fit_transform(pd.Series([symbol]*len(df15)))

    # 3) Última linha completa
    df15.dropna(inplace=True)
    if df15.empty:
        print(f"[IA_MODEL] {symbol} SKIP ENTRY: df15.dropna() resultou vazio")
        logging.info(f"[IA_MODEL] {symbol} SKIP ENTRY: df15.dropna() resultou vazio")
        return False
    feat = df15.iloc[[-1]]
    # Ensure all 57 features are present (fill missing with NaN)
    feat = ensure_all_features(feat, feature_columns)
    # Align columns to scaler.feature_names_in_ (robust to order and missing/extra columns)
    expected_features = list(scaler.feature_names_in_)
    X_feat = feat.reindex(columns=expected_features).astype('float32')
    # Debug logging for feature alignment
    logging.info(f"[DEBUG][IA_MODEL] {symbol} X_feat shape: {X_feat.shape}")
    logging.info(f"[DEBUG][IA_MODEL] {symbol} X_feat columns: {list(X_feat.columns)}")
    logging.info(f"[DEBUG][IA_MODEL] {symbol} Expected features: {list(scaler.feature_names_in_)}")
    logging.info(f"[DEBUG][IA_MODEL] {symbol} Any NaNs: {X_feat.isnull().any().any()}")
    # Check for missing or extra features
    missing = set(expected_features) - set(X_feat.columns)
    extra = set(X_feat.columns) - set(expected_features)
    if missing or extra:
        logging.error(f"[IA_MODEL] {symbol} Feature mismatch: missing={missing}, extra={extra}")
        return False
    # 5) Escala e prevê com o regressor
    Xs = scaler.transform(X_feat)
    pred_return = float(xgb_reg.predict(DMatrix(Xs))[0])
    logging.info(f"[IA_MODEL] {symbol} pred_return={pred_return:.6f}")
    # Record prediction for later evaluation
    bar_time = feat.index[-1]  # pd.Timestamp of the bar used for prediction
    record_prediction(symbol, bar_time, pred_return)
    if pred_return < IA_REG_THRESHOLD:
        logging.info(f"[IA_MODEL] {symbol} SKIP ENTRY: pred_return={pred_return:.6f} < {IA_REG_THRESHOLD}")
        return False
    # Envia mensagem no Telegram quando IA retornar True
    send_telegram(f"[IA_MODEL] {symbol} IA-model TRUE! pred_return={pred_return:.6f}")
    return True

# Lista de features exata usada pelo classificador (MUST match scaler)


def ensure_all_features(feat, feature_list):
    """
    Ensures all features in feature_list are present in feat DataFrame.
    Fills missing columns with np.nan and logs a warning if any are missing.
    """
    import numpy as np
    missing = [f for f in feature_list if f not in feat.columns]
    if missing:
        for f in missing:
            feat[f] = np.nan
        logging.warning(f"[FEATURE ALIGNMENT] Missing features filled with NaN: {missing}")
    # Reorder columns
    feat = feat.reindex(columns=feature_list)
    return feat

def ia_model_classifier_validation(symbol: str, threshold: float = 0.6) -> bool:
    """
    Valida a entrada usando o classificador stacked_clf.
    Retorna True se a probabilidade de alta for maior que o threshold.
    """
    logging.info(f"[IA_CLASSIFIER] {symbol} chamada para validação (threshold={threshold})")
    try:
        # 1. Busca candles e calcula features (igual ao pipeline do regressor)
        df1m = fetch_klines_df(symbol, Client.KLINE_INTERVAL_1MINUTE, limit=3000)
        if df1m.shape[0] < 60:
            logging.info(f"[IA_CLASSIFIER] {symbol} SKIP: menos de 60 candles 1m disponíveis")
            return False
        df1m = df1m[['open','high','low','close','volume']].sort_index()
        df15 = df1m.resample('15min').agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum'
        }).dropna()
        # FULL feature engineering block (identical to regressor, for 68 features)
        df15['return']       = df15['close'].pct_change()
        df15['price_change'] = df15['close'] - df15['open']
        df15['volatility']   = df15['return'].rolling(14).std()
        df15['direction']    = np.sign(df15['return']).fillna(0)
        df15['sma_20']  = df15['close'].rolling(20).mean()
        df15['ema_50']  = df15['close'].ewm(span=50,  adjust=False).mean()
        df15['ema_9']   = df15['close'].ewm(span=9,   adjust=False).mean()
        df15['ema_200'] = df15['close'].ewm(span=200, adjust=False).mean()
        from ta.trend      import ADXIndicator
        from ta.momentum   import RSIIndicator, StochasticOscillator, TSIIndicator
        from ta.volume     import MFIIndicator
        from ta.volatility import BollingerBands
        df15['adx_14']   = ADXIndicator(df15['high'], df15['low'], df15['close'], window=14).adx()
        df15['rsi_14']   = RSIIndicator(df15['close'], window=14).rsi()
        sto              = StochasticOscillator(df15['high'], df15['low'], df15['close'], window=14, smooth_window=3)
        df15['sto_k']    = sto.stoch()
        df15['sto_d']    = sto.stoch_signal()
        df15['macd']     = df15['close'].ewm(span=12, adjust=False).mean() - df15['close'].ewm(span=26, adjust=False).mean()
        df15['macd_sig'] = df15['macd'].ewm(span=9, adjust=False).mean()
        df15['roc_10']   = df15['close'].pct_change(10)
        df15['tsi_25']   = TSIIndicator(df15['close'], window_slow=25, window_fast=13).tsi()
        hl = df15['high'] - df15['low']
        hc = (df15['high'] - df15['close'].shift()).abs()
        lc = (df15['low']  - df15['close'].shift()).abs()
        df15['atr_14']   = pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(14).mean()
        bb              = BollingerBands(df15['close'], window=20, window_dev=2)
        df15['bb_width']     = bb.bollinger_hband() - bb.bollinger_lband()
        df15['bb_percent_b'] = bb.bollinger_pband()
        df15['dc_width']     = df15['high'].rolling(20).max() - df15['low'].rolling(20).min()
        df15['vol_ma_20']    = df15['volume'].rolling(20).mean()
        df15['vol_ratio_20'] = df15['volume'] / df15['vol_ma_20']
        df15['obv']          = (np.sign(df15['close'].diff()) * df15['volume']).cumsum()
        df15['vpt']          = (df15['close'].pct_change() * df15['volume']).cumsum()
        for p in [5,10,20,60]:
            df15[f'ret_{p}']     = df15['close'].pct_change(p)
            df15[f'logret_{p}']  = np.log(df15['close']/df15['close'].shift(p))
            df15[f'roll_std_{p}']= df15['return'].rolling(p).std()
            df15[f'roll_skew_{p}']= df15['return'].rolling(p).skew()
        # Add roll_std_6 for compatibility with old scaler
        df15['roll_std_6'] = df15['return'].rolling(6).std()
 
        # Candle anatomy features
        df15['upper_shadow'] = df15['high'] - df15[['close','open']].max(axis=1)
        df15['lower_shadow'] = df15[['close','open']].min(axis=1) - df15['low']
        df15['body_size']    = (df15['close'] - df15['open']).abs()

        # Money Flow Index
        df15['mfi_14'] = MFIIndicator(df15['high'], df15['low'], df15['close'], df15['volume'], window=14).money_flow_index()

        # 5-min return alias (for return_5min)
        df15['return_5min'] = df15['close'].pct_change(5)

        df15['minute']            = df15.index.minute
        df15['hour']              = df15.index.hour
        df15['dayofweek']         = df15.index.dayofweek
        df15['dayofmonth']        = df15.index.day
        df15['month']             = df15.index.month
        df15['year']              = df15.index.year
        df15['day']               = df15.index.day
        df15['weekday']           = df15.index.weekday
        df15['is_month_end']      = df15.index.is_month_end.astype(int)
        df15['is_month_start']    = df15.index.is_month_start.astype(int)
        df15['is_quarter_end']    = df15.index.is_quarter_end.astype(int)
        df15['mins_since_daystart']= df15.index.hour*60 + df15.index.minute
        df15['hour_sin']          = np.sin(2*np.pi*df15['hour']/24)
        df15['hour_cos']          = np.cos(2*np.pi*df15['hour']/24)
        df15['dow_sin']           = np.sin(2*np.pi*df15['dayofweek']/7)
        df15['dow_cos']           = np.cos(2*np.pi*df15['dayofweek']/7)
        df15['symbol_id'] = LabelEncoder().fit_transform(pd.Series([symbol]*len(df15)))
        # 3) Última linha completa
        df15.dropna(inplace=True)
        if df15.empty:
            logging.info(f"[IA_CLASSIFIER] {symbol} SKIP: df15.dropna() resultou vazio")
            return False

        feat = df15.iloc[[-1]]

        # --- Alinhamento seguro das features para o classificador ---
        # 1. Carrega a lista de features do classificador (68 features esperadas pelo scaler)
        FEATURE_CLASSIFIER_TXT = os.path.join(MODEL_DIR, 'feature_classifier.txt')
        if os.path.exists(FEATURE_CLASSIFIER_TXT):
            with open(FEATURE_CLASSIFIER_TXT) as f:
                feature_columns_classifier = [line.strip() for line in f if line.strip()]
            logging.info(f"[IA_CLASSIFIER] Usando features de feature_classifier.txt ({len(feature_columns_classifier)} features)")
        else:
            # Fallback: tenta pegar do scaler salvo, depois genérico
            if hasattr(scaler_classifier, 'feature_names_in_'):
                feature_columns_classifier = list(scaler_classifier.feature_names_in_)
                logging.warning(f"[IA_CLASSIFIER] feature_classifier.txt não encontrado, usando feature_names_in_ do scaler ({len(feature_columns_classifier)} features)")
            else:
                feature_columns_classifier = [f'f{i}' for i in range(scaler_classifier.n_features_in_)]
                logging.warning(f"[SCALER_CLASSIFIER] feature_names_in_ not found, usando nomes genéricos: {feature_columns_classifier}")

        # 2. Garante alinhamento e ordem das features do scaler (68)
        X_feat = feat.reindex(columns=feature_columns_classifier).astype('float32')

        # 3. Robustez: checa por features faltantes/NaN
        missing = X_feat.columns[X_feat.isnull().all()]
        if len(missing) > 0:
            logging.warning(f"[IA_CLASSIFIER] Features presentes no modelo mas ausentes do cálculo: {missing.tolist()}")
        if X_feat.isnull().any().any():
            nan_cols = X_feat.columns[X_feat.isnull().any()].tolist()
            logging.warning(f"[IA_CLASSIFIER] Features com NaN: {nan_cols}")
            return False

        # 4. Checagem de shape: scaler espera 68 features
        if hasattr(scaler_classifier, 'n_features_in_') and scaler_classifier.n_features_in_ != len(feature_columns_classifier):
            msg = f"[IA_CLASSIFIER ERROR] {symbol}: Feature shape mismatch, scaler espera: {scaler_classifier.n_features_in_}, alinhadas: {len(feature_columns_classifier)}"
            logging.error(msg)
            send_telegram(msg)
            return False

        # Debug logging antes do scaler
        logging.info(f"[DEBUG][IA_CLASSIFIER] {symbol} X_feat shape (pré-scaler): {X_feat.shape}")
        logging.info(f"[DEBUG][IA_CLASSIFIER] {symbol} X_feat columns (pré-scaler): {list(X_feat.columns)}")

        # 5. Escala as features (agora shape deve ser [1, 68])
        Xs = scaler_classifier.transform(X_feat)

        # 6. Reduz para as features reais do modelo (tipicamente 57, ordem certa)
        if hasattr(stacked_clf, 'feature_names_in_'):
            model_features = list(stacked_clf.feature_names_in_)
            Xs_df = pd.DataFrame(Xs, columns=feature_columns_classifier)
            Xs_final = Xs_df.reindex(columns=model_features).astype('float32')
            logging.info(f"[DEBUG][IA_CLASSIFIER] {symbol} Xs_final columns (modelo): {list(Xs_final.columns)}")
        else:
            Xs_final = Xs[:, :stacked_clf.n_features_in_]
            logging.warning(f"[IA_CLASSIFIER] feature_names_in_ não encontrado no modelo, cortando colunas.")

        # Checa shape final antes da predição
        if Xs_final.shape[1] != stacked_clf.n_features_in_:
            msg = f"[IA_CLASSIFIER ERROR] {symbol}: Feature shape mismatch FINAL, expected: {stacked_clf.n_features_in_}, got {Xs_final.shape[1]}"
            logging.error(msg)
            send_telegram(msg)
            return False

        # 7. Predição!
        proba = stacked_clf.predict_proba(Xs_final)[0][1]
        logging.info(f"[IA_CLASSIFIER] {symbol} prob_up={proba:.3f} (threshold={threshold})")
        send_telegram(f"[IA_CLASSIFIER] {symbol} prob_up={proba:.3f} (threshold={threshold})")
        return proba > threshold


    except Exception as e:
        logging.error(f"[IA_CLASSIFIER ERROR] {symbol}: {e}")
        return False
# ─── POSITION MANAGEMENT & MAIN LOOP ─────────────────────────────────────────────

def sync_positions_with_binance():
    """
    Queries Binance spot balances. If a base‐asset balance > dust threshold exists, 
    it populates positions[s] if not already present, using a structure analogous 
    to what the bot uses for new positions.
    """
    for symbol in PAIRS:
        base = symbol.replace("USDC","")
        try:
            bal = client.get_asset_balance(asset=base)
            free_qty = Decimal(bal["free"]) if bal and bal.get("free") else Decimal('0')
            rules = get_symbol_rules(symbol)
            # Check if it's dust or no asset:
            is_dust = (free_qty > 0) and ((free_qty < rules['min_qty']) or ((free_qty * Decimal(str(client.get_symbol_ticker(symbol=symbol)["price"]))) < rules['min_notional']))
            if not is_dust and free_qty > 0:
                if positions.get(symbol) is None:
                    # Create an “imported” position skeleton
                    entry_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
                    qty         = float(free_qty)
                    positions[symbol] = {
                        "side": "long",
                        "entry_price": entry_price,
                        "quantity":    qty,
                        "sl":          entry_price * (1 - SL_PCT_LONG),
                        "tp":          entry_price * (1 + TP_PCT),
                        "be_trigger":  entry_price * (1 + BE_PCT),
                        "be_moved":    False,
                        "risk":        None,
                        "partial_taken": False
                    }
                    position_times[symbol] = datetime.now(timezone.utc)
                    logging.info(f"[SYNC POSITION] {symbol} imported with qty={qty:.6f} @ {entry_price:.4f}")
                    save_state()
            else:
                if positions.get(symbol) is not None:
                    positions[symbol] = None
                    position_times[symbol] = None
                    logging.info(f"[SYNC] Cleared position for {symbol} (dust or no balance).")
        except Exception as e:
            logging.error(f"[SYNC ERROR] {symbol}: {e}")
            traceback.print_exc()

def main_loop():
    """
    This is the heart of the bot. It loops continuously, doing:
      1) Sync any externally placed spot positions with our local state.
      2) For each symbol, fetch 15m & 1h data, compute indicators, decide “math‐signal.”
      3) Fetch 1m→15m features, scale and ask IA model → decide “IA‐signal.”
      4) If no position is open, and both signals are True, place a market BUY per our risk sizing.
      5) If there IS a position, manage it: partial exit, breakeven move, full TP/SL, record PnL.
      6) Sleep a short while, then repeat.
    """
    global current_equity, equity_high

    while True:
        try:
            # Step A: Sync balances → keep local state consistent
            sync_positions_with_binance()
            # Evaluate predictions for completed bars
            evaluate_predictions()
            # Step B: For each symbol, attempt entry or manage existing pos
            for symbol in PAIRS:
                try:
                    # Fetch 15m and 1h data once up front to speed things up
                    df15 = fetch_klines_df(symbol, Client.KLINE_INTERVAL_15MINUTE, limit=100)
                    df1h = fetch_klines_df(symbol, Client.KLINE_INTERVAL_1HOUR, limit=50)
                    if df15.shape[0] < 50 or df1h.shape[0] < 30:
                        continue

                    # Skip pairs with only stale candles (older than 2 days)
                    from datetime import datetime, timezone, timedelta
                    now_utc = datetime.now(timezone.utc)
                    if not df15.empty and (now_utc - df15.index[-1]) > timedelta(days=2):
                        logging.info(f"[{symbol}] SKIP: last 15m candle older than 2 days ({df15.index[-1]})")
                        continue
                    if not df1h.empty and (now_utc - df1h.index[-1]) > timedelta(days=2):
                        logging.info(f"[{symbol}] SKIP: last 1h candle older than 2 days ({df1h.index[-1]})")
                        continue

                    # Update equity_high & dynamic risk:
                    if current_equity <= 0:
                        trade_risk = BASE_RISK
                    else:
                        drawdown = (current_equity / equity_high) - 1.0
                        if drawdown < -DD_STOP_PCT:
                            trade_risk = 0.005
                        elif current_equity > INITIAL_CAPITAL * 1.5:
                            trade_risk = 0.005
                        elif current_equity > INITIAL_CAPITAL * 1.2:
                            trade_risk = 0.015
                        else:
                            trade_risk = BASE_RISK

                    state = positions.get(symbol)

                    # ─── ENTRY LOGIC ───
                    if state is None:
                        if not combined_entry_check(symbol):
                            continue

                        price = float(client.get_symbol_ticker(symbol=symbol)["price"])
                        entry_price = price
                        sl_price    = entry_price * (1 - SL_PCT_LONG)
                        tp_price    = entry_price * (1 + TP_PCT)
                        be_trigger  = entry_price * (1 + BE_PCT)

                        # Cálculo de posição baseado em risco
                        if current_equity <= 0:
                            risk_amount = INITIAL_CAPITAL * trade_risk
                        else:
                            risk_amount = current_equity * trade_risk
                        position_size = risk_amount / max(entry_price - sl_price, 1e-8)

                        # Verificar saldo USDC e ajustes de tamanho
                        bal = client.get_asset_balance(asset="USDC")
                        usdc_free = float(bal["free"]) if bal and bal.get("free") else 0.0
                        if usdc_free >= 50:
                            max_cost = usdc_free * 0.5 * 0.99
                        else:
                            max_cost = usdc_free * 0.99
                        cost = position_size * entry_price
                        if cost > max_cost:
                            cost = max_cost
                            position_size = cost / entry_price

                        min_trade = MIN_TRADE_BY_PAIR.get(symbol, MIN_USDC_TRADE)
                        if cost < min_trade:
                            logging.info(f"[{symbol}] SKIP ENTRY: cost {cost:.2f} < minimum {min_trade:.2f}")
                            continue

                        # Efetivar ordem de compra
                        try:
                            order = client.order_market_buy(symbol=symbol, quoteOrderQty=round(cost,2))
                            fills = order.get('fills', [])
                            qty_filled = sum(float(fill["qty"]) for fill in fills)
                            if qty_filled <= 0:
                                logging.error(f"[{symbol}] ORDER ERROR: nenhuma quantidade preenchida")
                                continue
                            avg_price = (
                                sum(float(fill["price"]) * float(fill["qty"]) for fill in fills) / qty_filled
                            )

                            # Armazenar posição
                            positions[symbol] = {
                                "side": "long",
                                "entry_price": avg_price,
                                "quantity": qty_filled,
                                "sl": avg_price * (1 - SL_PCT_LONG),
                                "tp": avg_price * (1 + TP_PCT),
                                "be_trigger": avg_price * (1 + BE_PCT),
                                "be_moved": False,
                                "risk": trade_risk,
                                "partial_taken": False
                            }
                            position_times[symbol] = datetime.now(timezone.utc)
                            register_trade("BUY", symbol, qty_filled, avg_price, "MATH+IA_REG+CLF+RF(2)")
                            send_telegram(f"🟢 BUY {symbol} qty={qty_filled:.6f} @ {avg_price:.4f}")
                            save_state()

                            logging.info(f"[{symbol}] ENTRY EXECUTED: qty={qty_filled:.6f} @ price={avg_price:.4f}")

                        except BinanceAPIException as e:
                            logging.error(f"[{symbol}] ORDER ERROR BUY: {e}")
                        except Exception as e:
                            logging.error(f"[{symbol}] ORDER ERROR BUY (other): {e}")
                            traceback.print_exc()


                    # ——— POSITION MANAGEMENT ———
                    else:
                        pos = state
                        entry_price = Decimal(str(pos['entry_price']))
                        qty         = Decimal(str(pos['quantity']))
                        side        = pos.get('side','long')
                        try:
                            price = float(client.get_symbol_ticker(symbol=symbol)["price"])
                        except Exception as e:
                            logging.error(f"[{symbol}] PRICE FETCH ERROR: {e}")
                            continue
                        current_price = Decimal(str(price))
                        rules = get_symbol_rules(symbol)

                        # 1) Partial exit logic
                        try:
                            if (not pos.get('partial_taken', False)) and (side=='long'):
                                partial_trigger = entry_price * (Decimal('1') + PARTIAL_PCT)
                                if current_price >= partial_trigger:
                                    # Sell 50%
                                    partial_qty = (qty * PARTIAL_SIZE).quantize(rules['step_size'], rounding=ROUND_DOWN)
                                    logging.info(f"[PARTIAL SELL] {symbol}: step_size={rules['step_size']}, min_qty={rules['min_qty']}, initial partial_qty={partial_qty}")
                                    if partial_qty >= rules['min_qty']:
                                        attempt_qty = partial_qty
                                        max_retries = 8
                                        retries = 0
                                        while attempt_qty >= rules['min_qty'] and retries < max_retries:
                                            # Always quantize to step_size
                                            attempt_qty = (attempt_qty // rules['step_size']) * rules['step_size']
                                            # Check available balance before each attempt
                                            base = symbol.replace("USDC","")
                                            bal = client.get_asset_balance(asset=base)
                                            avail = Decimal(bal["free"]) if bal and bal.get("free") else Decimal('0')
                                            if avail < rules['min_qty']:
                                                logging.warning(f"[PARTIAL SELL] {symbol}: available balance {avail} < min_qty {rules['min_qty']}, skipping.")
                                                break
                                            if attempt_qty > avail:
                                                logging.warning(f"[PARTIAL SELL] {symbol}: attempt_qty {attempt_qty} > available {avail}, adjusting down.")
                                                attempt_qty = (avail // rules['step_size']) * rules['step_size']
                                                if attempt_qty < rules['min_qty']:
                                                    break
                                            partial_qty_str = format(attempt_qty.normalize(), 'f').rstrip('0').rstrip('.')
                                            try:
                                                client.order_market_sell(symbol=symbol, quantity=partial_qty_str)
                                                pos['quantity']      = float(qty - attempt_qty)
                                                pos['sl']            = float(entry_price)   # move stop to breakeven
                                                pos['partial_taken'] = True
                                                profit = (current_price - entry_price) * attempt_qty
                                                current_equity += float(profit)
                                                register_trade("PARTIAL_SELL", symbol, float(attempt_qty), float(price), "Partial +0.8%")
                                                logging.info(f"[PARTIAL SELL] {symbol}: qty={partial_qty_str} succeeded.")
                                                break
                                            except BinanceAPIException as e:
                                                if 'LOT_SIZE' in str(e):
                                                    attempt_qty = (attempt_qty - rules['step_size']).quantize(rules['step_size'], rounding=ROUND_DOWN)
                                                    logging.warning(f"[PARTIAL SELL] {symbol}: LOT_SIZE error, retrying with qty={attempt_qty}")
                                                    retries += 1
                                                    continue
                                                elif 'insufficient balance' in str(e).lower():
                                                    logging.error(f"[PARTIAL SELL ERROR] {symbol}: Insufficient balance, aborting partial sell.")
                                                    break
                                                else:
                                                    logging.error(f"[PARTIAL SELL ERROR] {symbol}: {e}")
                                                    break
                                        else:
                                            logging.error(f"[PARTIAL SELL] {symbol}: Could not satisfy LOT_SIZE after retries. Final qty={attempt_qty}, step_size={rules['step_size']}, min_qty={rules['min_qty']}")
                        except (DecimalException, Exception) as e:
                            logging.error(f"[PARTIAL ERROR] {symbol}: {e}")

                        # 2) Breakeven move
                        if (not pos.get('be_moved', False)) and side=='long':
                            if current_price >= Decimal(str(pos['be_trigger'])):
                                pos['sl'] = float(entry_price)
                                pos['be_moved'] = True

                        # 3) Full take‐profit or stop‐loss
                        reason = None
                        sell_qty = float(pos['quantity'])
                        # Use real-time price for stop-loss/take-profit
                        try:
                            ticker = client.get_symbol_ticker(symbol=symbol)
                            realtime_price = Decimal(str(ticker['price']))
                        except Exception as e:
                            logging.error(f"[REALTIME PRICE ERROR] {symbol}: {e}")
                            realtime_price = current_price  # fallback to last candle close
                        if side=='long':
                            if realtime_price >= Decimal(str(pos['tp'])):
                                reason = 'Take Profit +2%'
                            elif realtime_price <= Decimal(str(pos['sl'])):
                                reason = 'Stop Loss 0.5%'

                        if reason:
                            try:
                                # Determine exact sell size (all remaining)
                                rules = get_symbol_rules(symbol)
                                step = rules['step_size']
                                sell_qty_dec = (Decimal(str(sell_qty)) / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step
                                base = symbol.replace("USDC","")
                                bal = client.get_asset_balance(asset=base)
                                avail = Decimal(bal["free"]) if bal and bal.get("free") else Decimal('0')
                                sell_qty_dec = min(sell_qty_dec, avail)
                                logging.info(f"[EXIT ORDER] {symbol}: step_size={step}, min_qty={rules['min_qty']}, initial sell_qty={sell_qty_dec}")
                                if sell_qty_dec <= 0:
                                    logging.error(f"[EXIT SKIP] {symbol}: no {base} to sell")
                                else:
                                    if sell_qty_dec < rules['min_qty']:
                                        logging.warning(f"[EXIT REJECT] {symbol}: qty {sell_qty_dec} < min_qty {rules['min_qty']}")
                                    else:
                                        notional_val = sell_qty_dec * current_price
                                        if notional_val < rules['min_notional']:
                                            logging.warning(f"[EXIT REJECT] {symbol}: notional {notional_val} < {rules['min_notional']}")
                                        else:
                                            attempt_qty = sell_qty_dec
                                            max_retries = 8
                                            retries = 0
                                            while attempt_qty >= rules['min_qty'] and retries < max_retries:
                                                # Always quantize to step_size and check for negative
                                                attempt_qty = (attempt_qty // step) * step
                                                if attempt_qty < rules['min_qty'] or attempt_qty <= 0:
                                                    break
                                                sell_qty_str = format(attempt_qty.normalize(), 'f').rstrip('0').rstrip('.')
                                                try:
                                                    order = client.order_market_sell(symbol=symbol, quantity=sell_qty_str)
                                                    logging.info(f"[EXIT ORDER] SOLD {sell_qty_str} {symbol} @ {current_price} ({reason})")
                                                    send_telegram(f"[SELL ATTEMPT] {symbol} qty={sell_qty_str} @ {price:.4f} SUCCESS ({reason})")
                                                    # Remove position
                                                    positions[symbol] = None
                                                    position_times[symbol] = None
                                                    profit = (current_price - entry_price) * attempt_qty
                                                    current_equity += float(profit)
                                                    if current_equity > equity_high:
                                                        equity_high = current_equity
                                                    register_trade("SELL", symbol, float(attempt_qty), float(price), reason)
                                                    send_telegram(f"🔴 SELL {symbol} qty={sell_qty_str} @ {price:.4f} | {reason}")
                                                    save_state()
                                                    break
                                                except BinanceAPIException as e:
                                                    send_telegram(f"[SELL ATTEMPT] {symbol} qty={sell_qty_str} @ {price:.4f} FAILED: {e}")
                                                    if 'LOT_SIZE' in str(e):
                                                        attempt_qty = (attempt_qty - step).quantize(step, rounding=ROUND_DOWN)
                                                        logging.warning(f"[EXIT ORDER] {symbol}: LOT_SIZE error, retrying with qty={attempt_qty}")
                                                        retries += 1
                                                        continue
                                                    else:
                                                        logging.error(f"[EXIT ERROR] {symbol}: {e}")
                                                        break
                                            else:
                                                logging.error(f"[EXIT ORDER] {symbol}: Could not satisfy LOT_SIZE after retries. Final qty={attempt_qty}, step_size={step}, min_qty={rules['min_qty']}")
                            except BinanceAPIException as e:
                                logging.error(f"[EXIT ERROR] {symbol}: {e}")
                                traceback.print_exc()
                            except Exception as e:
                                logging.error(f"[EXIT ERROR] {symbol} (other): {e}")
                                traceback.print_exc()

                except Exception as e:
                    logging.error(f"[PAIR LOOP ERROR] {symbol}: {e}")
                    traceback.print_exc()

                time.sleep(0.2)  # avoid hammering Binance

            # Repeat cycle after short delay
            time.sleep(2)

        except Exception as e:
            logging.error(f"[MAIN LOOP ERROR] {e}")
            traceback.print_exc()
            time.sleep(10)

# ─── ENTRY POINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load persisted state
    load_state()

    # 2) Initialize equity from account snapshot
    try:
        bal_usdc = client.get_asset_balance(asset="USDC")
        usdc_free = float(bal_usdc["free"]) if bal_usdc and bal_usdc.get("free") else 0.0
        total_val = usdc_free
        for symbol in PAIRS:
            base = symbol.replace("USDC","")
            bal_base = client.get_asset_balance(asset=base)
            qty = float(bal_base["free"]) if bal_base and bal_base.get("free") else 0.0
            if qty > 0:
                price = float(client.get_symbol_ticker(symbol=symbol)["price"])
                total_val += qty * price
        INITIAL_CAPITAL = total_val if total_val>0 else 0.0
    except Exception as e:
        logging.error(f"[INIT ERROR] could not fetch initial capital: {e}")
        INITIAL_CAPITAL = 0.0

    current_equity = INITIAL_CAPITAL
    equity_high    = INITIAL_CAPITAL

    # Start the main loop in a background thread
    threading.Thread(target=main_loop, daemon=True).start()

    # Keep the main thread alive indefinitely
    while True:
        time.sleep(1)