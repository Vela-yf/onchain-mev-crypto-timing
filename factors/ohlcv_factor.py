import pandas as pd 
import numpy as np
import vectorbt as vbt  

# ==========================================
# 1. 动量与趋势类 (Momentum & Trend)
# ==========================================

# ROC (Rate of Change)
ROC = vbt.IndicatorFactory(
    class_name="ROC",
    short_name="roc",
    input_names=["close"],
    param_names=["window"],
    output_names=["roc"]
).from_apply_func(lambda close, window: pd.Series(close.flatten()).pct_change(window).values)

# MA Deviation (乖离率)
MADev = vbt.IndicatorFactory(
    class_name="MADev",
    short_name="madev",
    input_names=["close"],
    param_names=["window"],
    output_names=["madev"]
).from_apply_func(lambda close, window: (pd.Series(close.flatten()) / pd.Series(close.flatten()).rolling(window).mean() - 1).values)

# MACD
def apply_macd(close, fast_window, slow_window, signal_window):  
    close_s = pd.Series(close.flatten())
    ema_fast = close_s.ewm(span=fast_window, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow_window, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    hist = macd - signal
    return macd.values, signal.values, hist.values

MACD = vbt.IndicatorFactory(
    class_name="MACD",
    short_name="macd_ext",
    input_names=["close"],
    param_names=["fast_window", "slow_window", "signal_window"],
    output_names=["macd", "signal", "hist"]
).from_apply_func(apply_macd)

# TSI (True Strength Index)
def apply_tsi(close, r=25, s=13):  # type: ignore
    close_s = pd.Series(close.flatten())
    diff = close_s.diff()
    ema1 = diff.ewm(span=r, min_periods=r).mean()
    ema2 = ema1.ewm(span=s, min_periods=s).mean()
    abs_diff = diff.abs()
    abs_ema1 = abs_diff.ewm(span=r, min_periods=r).mean()
    abs_ema2 = abs_ema1.ewm(span=s, min_periods=s).mean()
    return (100 * (ema2 / abs_ema2)).values

TSI = vbt.IndicatorFactory(
    class_name="TSI",
    short_name="tsi",
    input_names=["close"],
    param_names=["r", "s"],
    output_names=["tsi"]
).from_apply_func(apply_tsi)

# Slope (线性回归斜率)
def rolling_slope_py(close, window):  # type: ignore
    from numpy.lib.stride_tricks import sliding_window_view
    close_arr = close.flatten()
    if len(close_arr) < window:
        return np.full_like(close_arr, np.nan)
    
    y = sliding_window_view(close_arr, window_shape=window)
    x = np.arange(window)
    x_mean = x.mean()
    y_mean = y.mean(axis=1)
    
    x_centered = x - x_mean
    y_centered = y - y_mean[:, np.newaxis]
    
    cov = (y_centered * x_centered).sum(axis=1) / window
    var = (x_centered**2).sum() / window
    
    slope = cov / var
    
    res = np.full_like(close_arr, np.nan)
    res[window-1:] = slope
    return res

Slope = vbt.IndicatorFactory(
    class_name="Slope",
    short_name="slope",
    input_names=["close"],
    param_names=["window"],
    output_names=["slope"]
).from_apply_func(rolling_slope_py)

# ==========================================
# 2. 反转与均值回归类 (Mean Reversion)
# ==========================================

# Bollinger Band Width & %B
def apply_bbands_attr(close, window, stdev):  # type: ignore
    close_s = pd.Series(close.flatten())
    middle = close_s.rolling(window).mean()
    std = close_s.rolling(window).std(ddof=0)
    upper = middle + stdev * std
    lower = middle - stdev * std
    
    width = (upper - lower) / middle
    pct_b = (close_s - lower) / (upper - lower)
    return width.values, pct_b.values

BBandsAttr = vbt.IndicatorFactory(
    class_name="BBandsAttr",
    short_name="bbattr",
    input_names=["close"],
    param_names=["window", "stdev"],
    output_names=["width", "pct_b"]
).from_apply_func(apply_bbands_attr)

# CCI
def apply_cci(high, low, close, window):  # type: ignore
    tp = (pd.Series(high.flatten()) + pd.Series(low.flatten()) + pd.Series(close.flatten())) / 3
    cci = (tp - tp.rolling(window).mean()) / (0.015 * tp.rolling(window).std())
    return cci.values

CCI = vbt.IndicatorFactory(
    class_name="CCI",
    short_name="cci",
    input_names=["high", "low", "close"],
    param_names=["window"],
    output_names=["cci"]
).from_apply_func(apply_cci)

# Distance to High/Low
def apply_dist_hl(close, high, low, window):  # type: ignore
    dist_h = pd.Series(close.flatten()) / pd.Series(high.flatten()).rolling(window).max() - 1
    dist_l = pd.Series(close.flatten()) / pd.Series(low.flatten()).rolling(window).min() - 1
    return dist_h.values, dist_l.values

DistHL = vbt.IndicatorFactory(
    class_name="DistHL",
    short_name="disthl",
    input_names=["close", "high", "low"],
    param_names=["window"],
    output_names=["dist_h", "dist_l"]
).from_apply_func(apply_dist_hl)

# KDJ
def apply_kdj(high, low, close, n, m1, m2):  # type: ignore
    high_s, low_s, close_s = pd.Series(high.flatten()), pd.Series(low.flatten()), pd.Series(close.flatten())
    low_n, high_n = low_s.rolling(n).min(), high_s.rolling(n).max()
    rsv = (close_s - low_n) / (high_n - low_n) * 100
    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    return k.values, d.values, (3 * k - 2 * d).values

KDJ = vbt.IndicatorFactory(
    class_name="KDJ",
    short_name="kdj",
    input_names=["high", "low", "close"],
    param_names=["n", "m1", "m2"],
    output_names=["k", "d", "j"]
).from_apply_func(apply_kdj)

# ==========================================
# 3. 成交量与资金流类 (Volume & Liquidity)
# ==========================================

# OBV
def apply_obv(close, volume):  # type: ignore
    close_s = pd.Series(close.flatten())
    return (np.sign(close_s.diff()).fillna(0) * pd.Series(volume.flatten())).cumsum().values # type: ignore

OBV = vbt.IndicatorFactory(
    class_name="OBV",
    short_name="obv",
    input_names=["close", "volume"],
    output_names=["obv"]
).from_apply_func(apply_obv)

# VWAP Deviation
def apply_vwap_dev(close, high, low, volume):  # type: ignore
    tp = (pd.Series(high.flatten()) + pd.Series(low.flatten()) + pd.Series(close.flatten())) / 3
    vwap = (tp * pd.Series(volume.flatten())).cumsum() / pd.Series(volume.flatten()).cumsum()
    return (pd.Series(close.flatten()) / vwap - 1).values

VWAPDev = vbt.IndicatorFactory(
    class_name="VWAPDev",
    short_name="vwapdev",
    input_names=["close", "high", "low", "volume"],
    output_names=["vwapdev"]
).from_apply_func(apply_vwap_dev)

# CMF
def apply_cmf(high, low, close, volume, window):  # type: ignore
    high_s, low_s, close_s, vol_s = pd.Series(high.flatten()), pd.Series(low.flatten()), pd.Series(close.flatten()), pd.Series(volume.flatten())
    mfv = (( (close_s - low_s) - (high_s - close_s) ) / (high_s - low_s).replace(0, np.nan)) * vol_s
    cmf = mfv.rolling(window).sum() / vol_s.rolling(window).sum()
    return cmf.values

CMF = vbt.IndicatorFactory(
    class_name="CMF",
    short_name="cmf",
    input_names=["high", "low", "close", "volume"],
    param_names=["window"],
    output_names=["cmf"]
).from_apply_func(apply_cmf)

# VolSurge
def apply_vol_surge(volume, window):  # type: ignore
    vol_s = pd.Series(volume.flatten())
    return (vol_s / vol_s.rolling(window).mean()).values

VolSurge = vbt.IndicatorFactory(
    class_name="VolSurge",
    short_name="volsu",
    input_names=["volume"],
    param_names=["window"],
    output_names=["volsu"]
).from_apply_func(apply_vol_surge)

# MFI
def apply_mfi(high, low, close, volume, window):  # type: ignore
    tp = (pd.Series(high.flatten()) + pd.Series(low.flatten()) + pd.Series(close.flatten())) / 3
    mf = tp * pd.Series(volume.flatten())
    tp_diff = tp.diff()
    pos_mf = mf.where(tp_diff > 0, 0).rolling(window).sum()
    neg_mf = mf.where(tp_diff < 0, 0).rolling(window).sum()
    return (100 - (100 / (1 + pos_mf / neg_mf))).values

MFI = vbt.IndicatorFactory(
    class_name="MFI",
    short_name="mfi",
    input_names=["high", "low", "close", "volume"],
    param_names=["window"],
    output_names=["mfi"]
).from_apply_func(apply_mfi)

# AdvVolRatio
def apply_adv_vol_ratio(close, volume, window):  # type: ignore
    vol_s = pd.Series(volume.flatten())
    cond = pd.Series(close.flatten()).diff() > 0
    return (vol_s.where(cond, 0).rolling(window).sum() / vol_s.rolling(window).sum()).values

AdvVolRatio = vbt.IndicatorFactory(
    class_name="AdvVolRatio",
    short_name="avr",
    input_names=["close", "volume"],
    param_names=["window"],
    output_names=["avr"]
).from_apply_func(apply_adv_vol_ratio)

# ==========================================
# 4. 波动率类 (Volatility)
# ==========================================

# ATR
def apply_atr(high, low, close, window):
    high_s, low_s, close_s = pd.Series(high.flatten()), pd.Series(low.flatten()), pd.Series(close.flatten())
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # vbt uses wilder's smoothing (ewm with alpha=1/n)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    return atr.values

ATR = vbt.IndicatorFactory(
    class_name="ATR",
    short_name="atr_ext",
    input_names=["high", "low", "close"],
    param_names=["window"],
    output_names=["atr"]
).from_apply_func(apply_atr)

# GKVol
def apply_gk_vol(open_p, high, low, close, window):  # type: ignore
    log_hl = np.log(pd.Series(high.flatten()) / pd.Series(low.flatten()))**2
    log_co = np.log(pd.Series(close.flatten()) / pd.Series(open_p.flatten()))**2
    gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window).mean()**0.5
    return gk.values

GKVol = vbt.IndicatorFactory(
    class_name="GKVol",
    short_name="gkvol",
    input_names=["open", "high", "low", "close"],
    param_names=["window"],
    output_names=["gkvol"]
).from_apply_func(apply_gk_vol)

# RealizedVol
def apply_realized_vol(close, window):  # type: ignore
    rv = (pd.Series(close.flatten()).pct_change()**2).rolling(window).sum()**0.5
    return rv.values

RealizedVol = vbt.IndicatorFactory(
    class_name="RealizedVol",
    short_name="rvol",
    input_names=["close"],
    param_names=["window"],
    output_names=["rvol"]
).from_apply_func(apply_realized_vol)

# HLSpread
def apply_hl_spread(high, low, close):  # type: ignore
    return ((pd.Series(high.flatten()) - pd.Series(low.flatten())) / pd.Series(close.flatten())).values

HLSpread = vbt.IndicatorFactory(
    class_name="HLSpread",
    short_name="hls",
    input_names=["high", "low", "close"],
    output_names=["hls"]
).from_apply_func(apply_hl_spread)

# ==========================================
# 5. 截面与分布特征类 (Intraday Structure)
# ==========================================

def apply_dist_stats(close, window):  # type: ignore
    ret = pd.Series(close.flatten()).pct_change()
    return ret.rolling(window).skew().values, ret.rolling(window).kurt().values

DistStats = vbt.IndicatorFactory(
    class_name="DistStats",
    short_name="dstats",
    input_names=["close"],
    param_names=["window"],
    output_names=["skew", "kurt"]
).from_apply_func(apply_dist_stats)

def apply_drp(close, high, low, window):  # type: ignore
    close_s, high_s, low_s = pd.Series(close.flatten()), pd.Series(high.flatten()), pd.Series(low.flatten())
    drp = (close_s - low_s.rolling(window).min()) / (high_s.rolling(window).max() - low_s.rolling(window).min())
    return drp.values

DailyRangePos = vbt.IndicatorFactory(
    class_name="DailyRangePos",
    short_name="drp",
    input_names=["close", "high", "low"],
    param_names=["window"],
    output_names=["drp"]
).from_apply_func(apply_drp)

def apply_bar_struct(open_p, high, low, close):  # type: ignore
    o, h, l, c = pd.Series(open_p.flatten()), pd.Series(high.flatten()), pd.Series(low.flatten()), pd.Series(close.flatten()) # type: ignore
    hl = (h - l).replace(0, np.nan)
    shadow = ((h - np.maximum(o, c)) + (np.minimum(o, c) - l)) / hl
    body = np.abs(c - o) / hl
    return shadow.values, body.values # type: ignore

BarStructure = vbt.IndicatorFactory(
    class_name="BarStructure",
    short_name="barstruct",
    input_names=["open", "high", "low", "close"],
    output_names=["shadow_ratio", "body_ratio"]
).from_apply_func(apply_bar_struct)

def apply_pv_corr(close, volume, window):  # type: ignore
    return pd.Series(close.flatten()).pct_change().rolling(window).corr(pd.Series(volume.flatten()).pct_change()).values

PVCorr = vbt.IndicatorFactory(
    class_name="PVCorr",
    short_name="pvcorr",
    input_names=["close", "volume"],
    param_names=["window"],
    output_names=["pvcorr"]
).from_apply_func(apply_pv_corr)

# ==========================================
# 6. 进阶“Alpha 101”风格因子
# ==========================================

def apply_ts_rank_vol(volume, window):  # type: ignore
    from numpy.lib.stride_tricks import sliding_window_view
    vol_arr = volume.flatten()
    if len(vol_arr) < window:
        return np.full_like(vol_arr, np.nan)
    
    y = sliding_window_view(vol_arr, window_shape=window)
    last_elements = y[:, -1:]
    ranks = (y <= last_elements).sum(axis=1) 
    
    res = np.full_like(vol_arr, np.nan)
    res[window-1:] = ranks / window
    return res

TSRankVol = vbt.IndicatorFactory(
    class_name="TSRankVol",
    short_name="tsrvol",
    input_names=["volume"],
    param_names=["window"],
    output_names=["rank"]
).from_apply_func(apply_ts_rank_vol)

def apply_corr_cv(close, volume, window):  # type: ignore
    return pd.Series(close.flatten()).rolling(window).corr(pd.Series(volume.flatten())).values

CorrCV = vbt.IndicatorFactory(
    class_name="CorrCV",
    short_name="corrcv",
    input_names=["close", "volume"],
    param_names=["window"],
    output_names=["corr"]
).from_apply_func(apply_corr_cv)

DeltaVol = vbt.IndicatorFactory(
    class_name="DeltaVol",
    short_name="dvol",
    input_names=["volume"],
    param_names=["window"],
    output_names=["delta"]
).from_apply_func(lambda volume, window: pd.Series(volume.flatten()).diff(window).values)

LogVol = vbt.IndicatorFactory(
    class_name="LogVol",
    short_name="lvol",
    input_names=["volume"],
    output_names=["logv"]
).from_apply_func(lambda volume: np.log1p(pd.Series(volume.flatten())).values) # type: ignore
