import warnings
# import eli5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
# from eli5.sklearn import PermutationImportance
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


xgb.set_config(verbosity=0)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width', 10000)
# pd.set_option('display.max_columns', None)

# file
filepath = "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/心律失常01COPD__clean.csv"
picpath = "/Users/gyw/Desktop/COPD/pics/"
test_size = 0.1  # radio
train_csv = "/Users/gyw/Desktop/COPD/Process_Files/train.csv"
test_csv = "/Users/gyw/Desktop/COPD/Process_Files/test.csv"


value_features_label = "NLR,PLR,SII,SIRI,RCI".split(",")  # value_features


print("\n------------------******* 划分数据集*******-------------------")
data = pd.read_csv(filepath)
y = data["GOLD1-4"]
X = data.drop( ["GOLD1-4", 'COPD&PN', 'PAC', 'PVC',
                'RBBB', 'SVT', 'AF', 'LBBB', 'HF'], axis = 1 )
X=np.array(X)
names= X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#
# print("\n训练集形状", X_train.shape, y_train.shape)
# print(" 类别比例:\n", y_train.value_counts() / len(y_train))
# over_samples = SMOTE(sampling_strategy="auto", k_neighbors=5, n_jobs=3)
# X_train, y_train = over_samples.fit_resample(X_train, y_train)
# # 数据规范化
# ss = preprocessing.StandardScaler()
# X_train = ss.fit_transform(X_train)
# X_test = ss.fit_transform(X_test)

df1 = pd.DataFrame(X_train)
df1["GOLD"] = y_train
df2 = pd.DataFrame(X_test)
df2["GOLD"] = y_test
# df2.to_csv(train_csv, header=True, index=False)
# df2.to_csv(test_csv, header=True, index=False)
print("\n重抽后的训练集形状", X_train.shape, y_train.shape)
print("重抽后的类别比例:\n", pd.Series(y_train).value_counts() / len(y_train))
print("\n------------------*******XGBmodel*******-------------------")
model_xgb = XGBClassifier(
    learning_rate=0.1,  # 学习率
    n_estimators=1000,  # 树的个数
    max_depth=5,  # 树的最大深度
    min_child_weight=1,  # 叶子节点样本权重加和最小值sum(H)
    gamma=0,  # 节点分裂所需的最小损失函数下降值
    subsample=0.8,  # 样本随机采样作为训练集的比例
    colsample_bytree=0.8,  # 使用特征比例
    objective='multi:softmax',  # 损失函数(这里为多分类）
    num_class=4,  # 多分类问题类别数
    seed=1)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
model_xgb.fit(X_train, y_train)
y_test_predict = model_xgb.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test_predict, y_test)
print('confusion_matrix:\n', confusion_matrix)
# 可视化
plt.figure(figsize=(20, 15))

sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel('predict')
plt.ylabel('true')
plt.xticks(ticks=[0.5, 1.5], labels=['no', 'yes'])
plt.yticks(ticks=[0.5, 1.5], labels=['no', 'yes'])
plt.show()
acc = sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
recall = confusion_matrix[1, 1] / sum(confusion_matrix[:, 1])
precision = confusion_matrix[1, 1] / sum(confusion_matrix[1, :])
print('测试集上的准确率为{:.2%}'.format(acc))
print('测试集上的召回率为{:.2%}'.format(recall))
print('测试集上的精确率为{:.2%}'.format(precision))
# print('f1_score:{}'.format(2 * recall * precision / (recall + precision)))


print("\n------------------*******初始模型k折交叉验证*******-------------------")
thresholds = np.sort(model_xgb.feature_importances_)

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model_xgb, threshold=thresh, prefit=True)
    select_train_x = selection.transform(X_train)
    xgb_param = model_xgb.get_params()
    xgtrain = xgb.DMatrix(X_train, y_train)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model_xgb.get_params()[
                      'n_estimators'], nfold=5, metrics='mlogloss', early_stopping_rounds=50)
    score = cvresult['test-mlogloss-mean'].iloc[-1]
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" %
          (thresh, select_train_x.shape[1], score))

print("\n------------------*******网格搜索调参*******-------------------\n")
print("01 初始参数下:n_estimators=", model_xgb.get_params()['n_estimators'])
cvresult = xgb.cv(
    xgb_param,
    xgtrain,
    num_boost_round=xgb_param['num_parallel_tree'],
    nfold=5,
    metrics='mlogloss',
    early_stopping_rounds=50)
model_xgb.set_params(n_estimators=cvresult.shape[0])
print("\n02 根据交叉验证结果修改:n_estimators=", model_xgb.get_params()['n_estimators'])
param_test = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 8, 2)
}
gsearch = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_test,
    scoring='accuracy',
    n_jobs=-1,
    cv=10)
gsearch.fit(X_train, y_train)
print(
    "\n03 max_depth 和 min_weight 粗调:",
    "best_score",
    gsearch.best_score_,
    "best_params",
    gsearch.best_params_)
curr_max_depth = gsearch.best_params_['max_depth']
curr_min_child_weight = gsearch.best_params_['min_child_weight']
param_test = {
    'max_depth': [
        curr_max_depth - 1,
        curr_max_depth,
        curr_max_depth + 1],
    'min_child_weight': [
        curr_min_child_weight - 1,
        curr_min_child_weight,
        curr_min_child_weight + 1]}
gsearch = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_test,
    scoring='accuracy',
    n_jobs=-1,
    cv=5)
gsearch.fit(X_train, y_train)
print(
    "\n04 max_depth 和 min_weight精调:",
    "best_score",
    gsearch.best_score_,
    "best_params",
    gsearch.best_params_)

print(gsearch.best_score_)
model_xgb.set_params(max_depth=gsearch.best_params_['max_depth'])
model_xgb.set_params(min_child_weight=gsearch.best_params_['min_child_weight'])

param_test = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
gsearch = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_test,
    scoring='accuracy',
    n_jobs=-1,
    cv=5)
gsearch.fit(X_train, y_train)
print(
    "\n05 gamma调优:",
    "best_score",
    gsearch.best_score_,
    "best_params",
    gsearch.best_params_)

model_xgb.set_params(gamma=gsearch.best_params_['gamma'])
param_test = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
gsearch = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_test,
    scoring='accuracy',
    n_jobs=-1,
    cv=5)
gsearch.fit(X_train, y_train)
print("\n06 subsample 和 colsample_bytree调优:", "best_score",
      gsearch.best_score_, "best_params", gsearch.best_params_)

model_xgb.set_params(subsample=gsearch.best_params_['subsample'])
model_xgb.set_params(colsample_bytree=gsearch.best_params_['colsample_bytree'])
print("\n07 正则化参数/降低学习率并调优增加决策树个数: n_estimators=",
      model_xgb.get_params()['n_estimators'])
model_xgb.set_params(learning_rate=0.01)
cvresult = xgb.cv(
    model_xgb.get_xgb_params(),
    xgtrain,
    num_boost_round=1000,
    nfold=5,
    metrics='mlogloss',
    early_stopping_rounds=50)
model_xgb.set_params(n_estimators=cvresult.shape[0])
print("\n08 根据交叉验证结果修改:n_estimators=", model_xgb.get_params()['n_estimators'])

print("\n------------------*******调参后*******-------------------")
y_test_predict = model_xgb.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test_predict, y_test)
print('confusion_matrix:\n', confusion_matrix)

# 可视化
# plt.figure(figsize=(, 15))

sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
plt.xlabel('predict')
plt.ylabel('true')
plt.xticks(ticks=[0.5, 1.5], labels=['no', 'yes'])
plt.yticks(ticks=[0.5, 1.5], labels=['no', 'yes'])
plt.show()
#
acc = sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
recall = confusion_matrix[1, 1] / sum(confusion_matrix[:, 1])
precision = confusion_matrix[1, 1] / sum(confusion_matrix[1, :])

print('测试集上的准确率为{:.2%}'.format(acc))
print('测试集上的召回率为{:.2%}'.format(recall))
print('测试集上的精确率为{:.2%}'.format(precision))
print('f1_score:{}'.format(2 * recall * precision / (recall + precision)))

print("\n------------------******* SHAP解释模型*******-------------------\n")
model_modify = model_xgb.save_raw()[4:]
# model_xgb.save_raw = myfun
# SHAP计算
explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X_train)
print(shap_values)
# 特征统计值
shap.summary_plot(shap_values, X_train)

# SHAP值解释
shap.summary_plot(shap_values[1], X_train, max_display=15)
shap.summary_plot(shap_values[2], X_train, max_display=15)
shap.summary_plot(shap_values[0], X_train, max_display=15)
#
# 训练集第1个样本对于输出结果为1的SHAP解释
shap.force_plot(explainer.expected_value[1],
                shap_values[1][1, :], X_train.iloc[1, :])
# 统计图解释
names= ['Admissiontime', 'age', 'BMI', 'Smokingindex', 'Hypertension', 'Diabetes', 'WBC', 'LY', 'NLR', 'PLR', 'SII', 'SIRI', 'RCI', 'PH', 'PCO2',
        'PO2', 'LAC', 'FiO2', 'Emphysema']

shap.bar_plot(shap_values[1][1, :], feature_names=names)
# # 输出第1个样本的特征值和shap值
sample_explainer = pd.DataFrame()
# sample_explainer['feature'] = cols
# sample_explainer['feature_value'] = X_train[cols].iloc[1].values

# 单个特征与模型预测结果的关系


# perm = PermutationImportance(model_xgb, random_state=1).fit(X_train, y_train)
# eli5.show_weights(perm, feature_names=names)
# eli5.show_prediction(perm, X_test)