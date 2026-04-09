import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  
import numpy as np
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(r"D:\khaki\ultralytics-8.3.27\mussel\results2\move_dis_and_rate-2017-2018-revise2.csv")
target_columns = ["rate","GT"]
data = df[target_columns]

rate = df["rate"]
gt = df["GT"]
#计算皮尔逊系数
corr, p_value = stats.spearmanr(rate,gt)
print(f"斯皮尔曼相关系数r：{corr},p值{p_value}")
#计算相关系数矩阵画热图
# df = pd.DataFrame(data)
# corr_matrix = df.corr()
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
# plt.title("Correlation rate and GT")
# plt.show()


#计算均方根误差（RMSE）和平均绝对误差(MAE)
mse = mean_squared_error(rate,gt)
rmse  = np.sqrt(mse)
print(f"均方根误差（RMSE）{rmse}")
print(f"方根误差（MSE）{mse}")
mae = mean_absolute_error(rate,gt)
print(f"平均绝对误差(MAE) {mae}")


#Bland-Altman Plots(一致性评价)
diff = rate -gt
mean_bias = np.mean(diff)
std_diff = np.std(diff)
loa_upper = mean_bias + 1.96 * std_diff
loa_lower = mean_bias - 1.96 * std_diff

print(f"Mean Bias: {mean_bias}")
print(f"SD of differences {std_diff}")
print(f"95% LoA: [{loa_lower}, {loa_upper}]")

