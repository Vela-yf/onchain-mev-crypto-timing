import pandas as pd
import numpy as np
import vectorbt as vbt
import sys
import os
import pdb

# 确保可以导入 factor 模块
sys.path.append(os.getcwd())

from ohlcv_factor import (
    ROC, MADev, MACD, TSI, Slope, 
    BBandsAttr, CCI, DistHL, KDJ,
    OBV, VWAPDev, CMF, VolSurge, MFI, AdvVolRatio,
    ATR, GKVol, RealizedVol, HLSpread,
    DistStats, DailyRangePos, BarStructure, PVCorr,
    TSRankVol, CorrCV, DeltaVol, LogVol
)

factor_dict = {
    "ROC": ROC,
    "MADev": MADev,
    "MACD": MACD,
    "TSI": TSI,
    #"Slope": Slope,
    "BBandsAttr": BBandsAttr,
    "CCI": CCI,
    "DistHL": DistHL,
    "KDJ": KDJ,
    "OBV": OBV,
    "VWAPDev": VWAPDev,
    "CMF": CMF,
    "VolSurge": VolSurge,
    "MFI": MFI,
    "AdvVolRatio": AdvVolRatio,
    "ATR": ATR,
    "GKVol": GKVol,
    "RealizedVol": RealizedVol,
    "HLSpread": HLSpread,
    "DistStats": DistStats,
    "DailyRangePos": DailyRangePos,
    "BarStructure": BarStructure,
    "PVCorr": PVCorr,
    "TSRankVol": TSRankVol,
    "CorrCV": CorrCV,
    "DeltaVol": DeltaVol,
    "LogVol": LogVol
}

data = pd.read_parquet("factor/price.parquet")
print(data.head(5))
print("************************")

def test_factors(name,factory):
    try:
            # 1. 获取该因子需要的输入参数名
            input_names = factory.input_names
            
            data_pool = {}
            data_pool["window"] = [6, 12, 48]
            # 2. 从 data 中提取对应的列
            # 注意：这里确保 data 包含 factory 需要的列名（如 'close', 'high' 等）
            for k in input_names:
                if k == "volume": data_pool[k] = data["q_volume"]
                else: data_pool[k] = data[k]
            
            # 3. 【关键修正】调用 .run() 而不是直接调用类
            indicator_run = factory.run(**data_pool)
            
            # 4. 获取输出
            # 如果你定义了多个 output_names，这里默认取第一个
            temp_result = pd.DataFrame()
            output_names = factory.output_names
            for out_name in output_names:
                result = getattr(indicator_run, out_name)

                # --- 关键修改：重命名列名 ---
                if isinstance(result, pd.DataFrame):
                    # 如果是多窗口结果，列名通常是 [6, 12, 48]
                    # 我们改为 "ROC_roc_6" 这种格式
                    result.columns = [f"{out_name}_{col}" for col in result.columns]
                else:
                    # 如果是 Series，给它个名字
                    result.name = f"{out_name}"
                # --------------------------
                assert isinstance(result, (pd.Series, pd.DataFrame)), f"{name} did not return a Series/DataFrame"
                print(f"{name} passed.")
                global factors
                temp_result = pd.concat([temp_result, result], axis=1, join="outer")
            return temp_result
    
    except Exception as e:
            print(f"{name} failed with error: {e}")

if __name__ == "__main__":
    factors = pd.DataFrame(index = data.index)
    for name, factory in factor_dict.items():
        new_factors = test_factors(name, factory)
        factors = pd.concat([factors, new_factors], axis=1, join="outer")
        print(factors.size)
    factors.to_parquet("factors.parquet")