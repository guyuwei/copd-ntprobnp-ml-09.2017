# 加载模型所需要的的包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 构造一个数据集，只包含一列数据，假如都是月薪数据，有些可能是错的
df = pd.DataFrame({'salary': [4, 1, 4, 5, 3, 6, 2, 5, 6, 2, 5, 7, 1, 8, 12, 33, 4, 7, 6, 7, 8, 55]})

# 构建模型 ,n_estimators=100 ,构建100颗树
model = IsolationForest(n_estimators=100,
						max_samples='auto',
						contamination=float(0.1),
						max_features=1.0)
# 训练模型
model.fit(df[['salary']])

# 预测 decision_function 可以得出 异常评分
df['scores'] = model.decision_function(df[['salary']])

#  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
df['anomaly'] = model.predict(df[['salary']])
print(df)