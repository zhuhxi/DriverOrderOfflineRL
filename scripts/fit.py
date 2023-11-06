import numpy as np
from scipy.stats import norm

# 假设你有一小段连续的数据
data = np.array([1.2, 1.5, 1.7, 1.4, 1.9])

# 使用最大似然估计拟合正态分布
estimated_mean, estimated_std = norm.fit(data)

# 打印估计的均值和标准差
print("Estimated mean:", estimated_mean)
print("Estimated standard deviation:", estimated_std)