from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

# pd.set_option('display.width', 10000)

# file
filepath = "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/心律失常01COPD__clean.csv"
cfile = "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/副本COPD合并心率失常1008.xlsx"
test_size = 0.2  # radio



value_features_label = "NLR,PLR,SII,SIRI,RCI".split(",")  # value_features

print("\n------------------******* 划分数据集*******-------------------")

data = pd.read_csv(filepath)

print(data.columns)


y = data["GOLD1-4"]
X = data.drop(["GOLD1-4", 'COPD&PN', 'PAC', 'PVC',
               'RBBB', 'SVT', 'AF', 'LBBB', 'HF'], axis=1)
X = data.drop( ["GOLD1-4"], axis = 1 )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

print("\n训练集形状", X_train.shape, y_train.shape)
print(" 类别比例:\n", y_train.value_counts() / len(y_train))
over_samples = SMOTE(sampling_strategy="minority", k_neighbors=3, n_jobs=1)

X_train, y_train = over_samples.fit_resample(X_train, y_train)

# 数据规范化xA
ss = preprocessing.StandardScaler()
ss_X_train = ss.fit_transform(X_train)
ss_X_test = ss.fit_transform(X_test)

dtc = DecisionTreeClassifier()
dtc.fit(ss_X_train, y_train)

predict_y = dtc.predict(ss_X_test)
print("分类决策树")
print("mean_squared_error:", mean_squared_error(y_test, predict_y))
print('accuracy_score：', accuracy_score(y_test, predict_y))


print("SVC")

svc = SVC()
svc.fit(ss_X_train, y_train)
predict_y = svc.predict(ss_X_test)
print("mean_squared_error:", mean_squared_error(y_test, predict_y))
print('accuracy_score：', accuracy_score(y_test, predict_y))
print("KNN")


knn = KNeighborsClassifier()
knn.fit(ss_X_train, y_train)
predict_y = knn.predict(ss_X_test)
print("mean_squared_error:", mean_squared_error(y_test, predict_y))
print('accuracy_score：', accuracy_score(y_test, predict_y))
print("AdaBoostClassifier")


ada = AdaBoostClassifier()
ada.fit(ss_X_train, y_train)
predict_y = ada.predict(ss_X_test)
print("mean_squared_error:", mean_squared_error(y_test, predict_y))
print('accuracy_score：', accuracy_score(y_test, predict_y))