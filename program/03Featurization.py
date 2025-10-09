# Todo 特征工程

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 支持中文显示

plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width', 10000)
# pd.set_option('display.max_columns', None)


filepath = "/Users/gyw/Desktop/Project/2022/张博超：XGBCOPD/COPD&NSCLC/CS/Process_Files/COPD_CS_Clean.csv"
picpath = "/Users/gyw/Desktop/COPD/pics/"
test_size = 0.2  # radio
train_csv = "/Users/gyw/Desktop/COPD/Process_Files/train.csv"
test_csv = "/Users/gyw/Desktop/COPD/Process_Files/test.csv"

test_size = 0.2
droplabels = ["id", "sex", "FEV1%Pred", "FVC%Pred", "FEV1pred"]


def boxpic(x, label, name):
	# TODO:查看特征变量的箱线图分布
	data = x.copy()
	columns = data.columns.tolist()
	dis_rows = 8
	dis_cols = 5
	plt.figure(figsize=(4 * dis_cols, 4 * dis_rows))
	for i in range(len(columns)):
		ax = plt.subplot(dis_rows, dis_cols, i + 1)
		ax = sns.boxplot(data=data[columns[i]], orient="v", width=0.5)
		ax = plt.xlabel(columns[i], fontsize=20)
	plt.tight_layout()
	plt.title(label + ":" + name)
	plt.savefig(picpath + '特征变量箱线图:' + label + name + ".png", dpi=300)
	plt.show()


def variation_cal(x):
	# TODO:计算异众比例
	print("计算异众比例")
	variation_ratio_s = 0.01
	data = x.copy()
	for col in data.columns:
		df_count = data[col].value_counts()
		kind = df_count.index[0]
		variation_ratio = 1 - (df_count.iloc[0] / len(data[col]))
		if variation_ratio < variation_ratio_s:
			print(f'\n{col} 最多的取值为{kind}，异众比例为{round(variation_ratio, 4)}')


def kdepic(x, y, label):
	# TODO:，训练集和测试集X分布对比
	x_train = x.copy()
	x_test = y.copy()
	columns = x_train.columns.tolist()
	dis_rows = 8
	dis_cols = 5
	plt.figure(figsize=(5 * dis_cols, 5 * dis_rows))
	for i in range(len(columns)):
		ax = plt.subplot(dis_rows, dis_cols, i + 1)
		ax = sns.kdeplot(x_train[columns[i]], color="Red", shade=True)
		ax = sns.kdeplot(x_test[columns[i]], color="Blue",
						 warn_singular=False, shade=True)
		ax.set_xlabel(columns[i], fontsize=20)
		ax.set_ylabel("Frequency", fontsize=18)
		ax = ax.legend(["train", "test"])
		plt.tight_layout()
	plt.title(label + u'特征变量核密度图')
	plt.savefig(picpath + '特征变量核密度图:' + label + ".png", dpi=300)
	plt.show()


def targetpic(x, label, name):
	# TODO: 查看y的分布
	target = x.copy()
	plt.figure(figsize=(10, 15), dpi=128)
	plt.subplot(3, 1, 1)
	target.plot.box()
	plt.subplot(3, 1, 2)
	target.plot.hist()
	plt.subplot(3, 1, 3)
	target.plot.kde()
	sns.kdeplot(target, color='Red', shade=True)

	plt.tight_layout()
	plt.title(label + ":" + name)


def isolation_forest(x):
	# TODO: 查看离群值
	print("计算离群值")
	data = x.copy()
	for col in data.columns:
		df=data[col]
		print(col)
		# 构建模型 ,n_estimators=100 ,构建100颗树
		model = IsolationForest(n_estimators=100,
								max_samples='auto',
								contamination=float(0.1),
								max_features=1.0)
		model.fit(df[col])

		# 预测 decision_function 可以得出 异常评分
		scores = model.decision_function(df[col])

		#  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
		anomaly = model.predict(df[col])
		df1 = pd.DataFrame()
		df1["scores"] = scores
		df1["anomaly"] = anomaly
		df1[col] = df[col]
		print(df1)


if __name__ == '__main__':
	print("\n------------------*******特征工程*******-------------------")
	print("\n----------------*******01 数据读取*******-------------------\n")
	data = pd.read_csv(filepath)
	print(data.info())

	print("\n------------------*******02 数据分析*******-------------------\n")
	y = data["GOLD"]
	X = data.drop(["GOLD"], axis=1)
	# boxpic(X, "GOLD", "X_train")
	variation_cal(X)

	print("\n------------------*******03 划分数据集分析*******-------------------\n")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size)
	print(" \n类别比例:\n", y_train.value_counts() / len(y_train))
	# 平衡训练数据集
	over_samples = SMOTE(
		sampling_strategy="minority",
		k_neighbors=2,
		n_jobs=1)
	over_samples_X, over_samples_y = over_samples.fit_resample(
		X_train, y_train)

	print(
		"\n 重抽后的类别比例:\n",
		pd.Series(over_samples_y).value_counts() /
		len(over_samples_y))
	print("重抽后的训练集形状", over_samples_X.shape, over_samples_y.shape)
	targetpic(y, "GOLD", "y_train")