import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 处理有效天数
def effective_days(factors, target_returns, window=144, threshold=0.1):
    # 1. 计算滚动 IC (使用 apply 处理多列)
    min_periods = int(window * 0.8)
    rolling_ic = factors.apply(lambda col: col.rolling(window=window, min_periods=min_periods).corr(target_returns))
    
    # 2. 判定有效性
    is_effective = rolling_ic.abs() > threshold
    
    # 3. 核心扁平化逻辑：
    # 找出发生变化的时刻（False变True 或 True变False）
    diff = is_effective != is_effective.shift()
    
    # 累加变化点，生成“块” ID (Group ID)
    group_ids = diff.cumsum()
    
    # 更简洁的 apply 方式处理 counts：
    # 遍历每一列，利用 group_ids 进行分组计数，最后滤掉无效（False）的部分
    persistence_df = is_effective.copy()
    for col in is_effective.columns:
        # 仅对 is_effective 为 True 的区间保留计数值，False 的区间设为 0
        persistence_df[col] = is_effective[col].groupby(group_ids[col]).transform('sum')
        persistence_df[col] = persistence_df[col].where(is_effective[col], 0)
        
    return persistence_df


# Jaccard 相似度
def get_jaccard_similarity(persistence_df):
    """
    严格基于原始数据计算因子间的 Jaccard 相似度矩阵
    输入: persistence_df (700000 rows, 60 columns)
    """
    # 1. 将有效性数据转为二值化矩阵 (0 或 1)
    # 只要 persistence >= 5 即判定为有效时刻
    A = (persistence_df >= 5).astype(np.int32).values
    
    # 2. 计算交集矩阵 (Intersection): A^T * A
    # 得到 (60, 60) 矩阵，每个元素表示两个因子同时有效的 bar 数
    intersection = np.dot(A.T, A)
    
    # 3. 获取每个因子的总有效数 (自交集)
    # intersection 的对角线即为每个因子各自有效的总计数
    self_sum = np.diag(intersection)
    
    # 4. 计算并集矩阵 (Union): Count(A) + Count(B) - Intersection(A, B)
    # 利用 NumPy 广播机制: (60, 1) + (1, 60) - (60, 60)
    union = self_sum[:, None] + self_sum[None, :] - intersection
    
    # 5. 计算 Jaccard = Intersection / Union
    # 排除分母为 0 的情况（即两个因子在 70w 行中从未生效）
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity_matrix = np.true_divide(intersection, union)
        similarity_matrix[union == 0] = 0
        
    # 返回 DataFrame 格式，保留因子名称
    return pd.DataFrame(similarity_matrix, index=persistence_df.columns, columns=persistence_df.columns)

# 谱聚类
def spectral_clustering(sim_matrix_df, n_clusters=10):
    """
    基于预计算的相似度矩阵进行谱聚类
    """
    # 提取矩阵数值
    adj_matrix = sim_matrix_df.values
    
    # 构建谱聚类模型
    # affinity='precomputed' 意味着输入必须是方阵相似度
    # assign_labels='discretize' 对初始聚类更鲁棒
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='discretize'
    )
    
    labels = model.fit_predict(adj_matrix)
    
    # 封装结果
    cluster_results = pd.Series(labels, index=sim_matrix_df.index, name='Cluster')
    return cluster_results


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_factor_ic_bar(factors: pd.Series, result: pd.Series, factor_name: str, result_name: str, bins=20):
    # 1. 数据对齐与准备
    data = pd.DataFrame({'factor': factors, 'result': result})
    
    # 使用 rank(method='first') 强制分组，确保每个数据都有位置
    data["rank"] = data["factor"].rank(method='first')
    
    # 2. 统一分组（使用 qcut 确保每组人数相等）
    # labels=False 返回 0, 1, 2... 这样的整数索引，作为横坐标
    data["group"] = pd.qcut(data["rank"], q=bins, labels=False)
    
    # 3. 计算统计量
    # 核心：计算每组的收益均值
    group_stats = data.groupby('group')['result'].mean()
    # 核心：计算每组的因子数值边界（用来做横轴标签）
    group_factor_range = data.groupby('group')['factor'].mean() 
    # 计算每组样本量（虽然 qcut 理论上是平均的，但画出来可以确认分布）
    group_counts = group_factor_range
    
    # 4. 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 横坐标使用 group 的整数索引：0, 1, 2, ..., bins-1
    x_axis = np.arange(bins)

    # 绘制左轴：因子分布（柱状图）
    ax1.bar(x_axis, group_counts, color='grey', alpha=0.3, 
            edgecolor='white', label = "Factor Value per Group")
    ax1.set_xlabel(f'{factor_name} Groups (From Low to High)')
    ax1.set_ylabel('Factor Value')

    # 5. 绘制右轴：收益率（折线图）
    ax2 = ax1.twinx()
    # 确保 x_axis 和 group_stats.values 长度一致
    ax2.plot(x_axis, group_stats, color='red', marker='o', alpha=0.4,
               linewidth=2, label=f'Mean {result_name} per Group')
    
    # 6. 优化 X 轴标签：显示该组因子的平均值
    # 每隔几个显示一个标签，防止太挤
    step = max(1, bins // 10)
    ax1.set_xticks(x_axis[::step])
    ax1.set_xticklabels([f"{val:.2f}" for val in group_factor_range.values[::step]])


    # 图例合并
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(f'{factor_name} Decile Analysis: Distribution vs. Mean {result_name}')
    plt.tight_layout()
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

def plot_factor_corr(df_factors, method='spearman'):
    """
    绘制因子相关性热力图
    method: 'pearson' 或 'spearman' (量化通常用后者，因为对异常值不敏感)
    """
    # 1. 计算相关系数矩阵
    corr_matrix = df_factors.corr(method=method)
    
    # 2. 设置绘图风格
    sns.set_theme(style="white")
    plt.figure(figsize=(12, 10))
    
    # 3. 绘制热力图
    # mask: 隐藏右上角重复的三角形，让视觉更聚焦
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    heatmap = sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True,           # 显示具体数值
        fmt=".2f",            # 保留两位小数
        cmap='vlag',          # 蓝红配色，0为白色，正负分明
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    
    plt.title(f'Factor Correlation Heatmap ({method.capitalize()})', fontsize=16)
    plt.show()
