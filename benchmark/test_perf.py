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

from ohlcv_factor import (
    ROC, MADev, MACD, TSI, Slope, BBandsAttr, CCI, DistHL, KDJ, OBV, VWAPDev, CMF,
    VolSurge, MFI, AdvVolRatio, ATR, GKVol, RealizedVol, HLSpread, DistStats, DailyRangePos,
    BarStructure, PVCorr, TSRankVol, CorrCV, DeltaVol, LogVol
)

def profile_factor(factor_cls, **kwargs):
    start = time.time()
    try:
        factor_cls.run(**kwargs)
        end = time.time()
        print(f"{factor_cls.__name__}: {end - start:.4f}s")
    except Exception as e:
        print(f"{factor_cls.__name__}: ERROR {e}")

if __name__ == '__main__':
    print("Profiling factors...")
    
    profile_factor(ROC, close=data['close'], window=14)
    profile_factor(MADev, close=data['close'], window=14)
    profile_factor(MACD, close=data['close'], fast_window=12, slow_window=26, signal_window=9)
    profile_factor(TSI, close=data['close'], r=25, s=13)
    profile_factor(Slope, close=data['close'], window=14)
    
    profile_factor(BBandsAttr, close=data['close'], window=20, stdev=2)
    profile_factor(CCI, high=data['high'], low=data['low'], close=data['close'], window=14)
    profile_factor(DistHL, close=data['close'], high=data['high'], low=data['low'], window=14)
    profile_factor(KDJ, high=data['high'], low=data['low'], close=data['close'], n=9, m1=3, m2=3)
    
    profile_factor(OBV, close=data['close'], volume=data['volume'])
    profile_factor(VWAPDev, close=data['close'], high=data['high'], low=data['low'], volume=data['volume'])
    profile_factor(CMF, high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], window=20)
    profile_factor(VolSurge, volume=data['volume'], window=14)
    profile_factor(MFI, high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], window=14)
    profile_factor(AdvVolRatio, close=data['close'], volume=data['volume'], window=14)
    
    profile_factor(ATR, high=data['high'], low=data['low'], close=data['close'], window=14)
    profile_factor(GKVol, open=data['open'], high=data['high'], low=data['low'], close=data['close'], window=14)
    profile_factor(RealizedVol, close=data['close'], window=14)
    profile_factor(HLSpread, high=data['high'], low=data['low'], close=data['close'])
    
    profile_factor(DistStats, close=data['close'], window=14)
    profile_factor(DailyRangePos, close=data['close'], high=data['high'], low=data['low'], window=14)
    profile_factor(BarStructure, open=data['open'], high=data['high'], low=data['low'], close=data['close'])
    profile_factor(PVCorr, close=data['close'], volume=data['volume'], window=14)
    
    profile_factor(TSRankVol, volume=data['volume'], window=14)
    profile_factor(CorrCV, close=data['close'], volume=data['volume'], window=14)
    profile_factor(DeltaVol, volume=data['volume'], window=14)
    profile_factor(LogVol, volume=data['volume'])
