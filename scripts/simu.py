import numpy as np
from scipy.stats import truncnorm

# 设置拟合区间的上下界
a, b = 2, 5

# 拟合并生成被截断的正态分布
mean, std = 3, 1
lower_bound = (a - mean) / std
upper_bound = (b - mean) / std
trunc_norm = truncnorm(lower_bound, upper_bound, loc=mean, scale=std)

# 生成采样数据
sample_size = 1000
samples = trunc_norm.rvs(size=sample_size)

print(samples)