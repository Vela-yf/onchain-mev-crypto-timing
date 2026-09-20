import pandas as pd
import numpy as np
import time

def generate_data(n=10000):
    np.random.seed(42)
    return {
        'open': np.random.rand(n) * 100 + 100,
        'high': np.random.rand(n) * 100 + 150,
        'low': np.random.rand(n) * 100 + 50,
        'close': np.random.rand(n) * 100 + 100,
        'volume': np.random.rand(n) * 1000
    }

data = generate_data()

# Original functions
def apply_macd(close, fast_window, slow_window, signal_window):  
    import vectorbt as vbt
    m = vbt.MACD.run(close, fast_window=fast_window, slow_window=slow_window, signal_window=signal_window)
    return m.macd.values, m.signal.values, m.hist.values # type: ignore

def rolling_slope_py(close, window):  
    close_s = pd.Series(close.flatten())
    def get_slope(y):  
        if len(y) < window or np.any(np.isnan(y)): return np.nan
        return np.polyfit(np.arange(len(y)), y, 1)[0]
    return close_s.rolling(window).apply(get_slope).values

def apply_ts_rank_vol(volume, window):  
    return pd.Series(volume.flatten()).rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).values

# Fast functions
def apply_macd_fast(close, fast_window, slow_window, signal_window):
    close_s = pd.Series(close.flatten())
    ema_fast = close_s.ewm(span=fast_window, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow_window, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    hist = macd - signal
    return macd.values, signal.values, hist.values

def rolling_slope_fast(close, window):
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

def apply_ts_rank_vol_fast(volume, window):
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

print("Testing MACD...")
start = time.time()
m1, s1, h1 = apply_macd(data['close'], 12, 26, 9)
print(f"Original: {time.time() - start:.4f}s")
start = time.time()
m2, s2, h2 = apply_macd_fast(data['close'], 12, 26, 9)
print(f"Fast: {time.time() - start:.4f}s")
# Due to vbt MACD using adjust=True maybe, we can check max diff
print("MACD max diff:", np.nanmax(np.abs(m1 - m2)))

print("\nTesting Slope...")
start = time.time()
slope1 = rolling_slope_py(data['close'], 14)
print(f"Original: {time.time() - start:.4f}s")
start = time.time()
slope2 = rolling_slope_fast(data['close'], 14)
print(f"Fast: {time.time() - start:.4f}s")
print("Slope max diff:", np.nanmax(np.abs(slope1 - slope2)))

print("\nTesting TSRankVol...")
start = time.time()
rank1 = apply_ts_rank_vol(data['volume'], 14)
print(f"Original: {time.time() - start:.4f}s")
start = time.time()
rank2 = apply_ts_rank_vol_fast(data['volume'], 14)
print(f"Fast: {time.time() - start:.4f}s")
print("Rank max diff:", np.nanmax(np.abs(rank1 - rank2)))
