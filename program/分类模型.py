import warnings
from sklearn import svm  # svm函数需要的
import pandas as pd
import numpy as np  # numpy科学计算库
from sklearn import model_selection
import matplotlib.pyplot as plt  # 画图的库
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
warnings.filterwarnings( "ignore" )
warnings.simplefilter( action = 'ignore', category = FutureWarning )


# ===========绘制ROC曲线================
def Draw_ROC ( list1, list2 ):
	
	fpr_model, tpr_model, thresholds = roc_curve( list1, list2, pos_label = 1 )
	roc_auc_model = auc( fpr_model, tpr_model )
	font = { 'family': 'Times New Roman', 'size': 12, }
	sns.set( font_scale = 1.2 )
	plt.rc( 'font', family = 'Times New Roman' )
	plt.plot( fpr_model, tpr_model, 'blue', label = 'AUC = %0.2f' % roc_auc_model )
	
	plt.legend( loc = 'lower right', fontsize = 12 )
	plt.plot( [0, 1], [0, 1], 'r--' )
	
	plt.ylabel( 'True  Positive Rate', fontsize = 14 )
	
	plt.xlabel( 'Flase  Positive Rate', fontsize = 14 )
	
	plt.show( )
	
	return


def modelclassfication ( X_train, X_test, y_train, y_test, k ):
	kfold = KFold( n_splits = k )
	print( "XGBmodel" )
	xgb = XGBClassifier(
			learning_rate = 0.1,  # 学习率
			n_estimators = 1000,  # 树的个数
			max_depth = 5,  # 树的最大深度
			min_child_weight = 1,  # 叶子节点样本权重加和最小值sum(H)
			gamma = 0,  # 节点分裂所需的最小损失函数下降值
			subsample = 0.8,  # 样本随机采样作为训练集的比例
			colsample_bytree = 0.8,  # 使用特征比例
			objective = 'multi:softmax',  # 损失函数(这里为多分类）
			num_class = 4,  # 多分类问题类别数
			seed = 1 )
	le = LabelEncoder( )
	y_train = le.fit_transform( y_train )
	xgb.fit( X_train, y_train )
	scores = cross_val_score( xgb, X_train, y_train, cv =kfold  )
	predict_y = xgb.predict( X_test )
	
	# 输出每折的准确率
	for i, score in enumerate( scores ):
		print( "Fold {}: {:.4f}".format( i + 1, score ) )
	# 输出平均准确率
	print( "Average Accuracy: {:.4f}".format( scores.mean( ) ) )
	print( "mean_squared_error:", mean_squared_error( y_test, predict_y ) )
	print( 'accuracy_score：', accuracy_score( y_test, predict_y ) )
	
	
	ss = preprocessing.StandardScaler( )
	X_train = ss.fit_transform( X_train )
	X_test = ss.fit_transform( X_test )
	
	print( "分类决策树" )
	
	dtc = DecisionTreeClassifier( )
	dtc.fit( X_train, y_train )

	scores = cross_val_score( dtc, X_train, y_train, cv =kfold )
	predict_y = dtc.predict( X_test )
	# 输出每折的准确率
	for i, score in enumerate( scores ):
		print( "Fold {}: {:.4f}".format( i + 1, score ) )
	# 输出平均准确率
	print( "Average Accuracy: {:.4f}".format( scores.mean( ) ) )
	print( "mean_squared_error:", mean_squared_error( y_test, predict_y ) )
	print( 'accuracy_score：', accuracy_score( y_test, predict_y ) )
	
	
	
	print( "SVC" )
	
	svc = SVC( )
	svc.fit(X_train, y_train )

	scores = cross_val_score( svc, X_train, y_train, cv = kfold )
	# 输出每折的准确率
	for i, score in enumerate( scores ):
		print( "Fold {}: {:.4f}".format( i + 1, score ) )
	# 输出平均准确率
	print( "Average Accuracy: {:.4f}".format( scores.mean( ) ) )
	predict_y = svc.predict( X_test )
	
	print( "mean_squared_error:", mean_squared_error( y_test, predict_y ) )
	print( 'accuracy_score：', accuracy_score( y_test, predict_y ) )
	
	print( "AdaBoostClassifier" )
	ada = AdaBoostClassifier( )
	ada.fit( X_train, y_train )
	
	scores = cross_val_score( ada, X_train, y_train, cv = kfold )
	# 输出每折的准确率
	for i, score in enumerate( scores ):
		print( "Fold {}: {:.4f}".format( i + 1, score ) )
	# 输出平均准确率
	print( "Average Accuracy: {:.4f}".format( scores.mean( ) ) )
	predict_y = ada.predict( X_test )
	print( "mean_squared_error:", mean_squared_error( y_test, predict_y ) )
	print( 'accuracy_score：', accuracy_score( y_test, predict_y ) )

#== == == == == == == 随机森林 == == == == == == == == =
# def RF ( X_train,y_train,X_test,y_test ):
	rf_classifier = RandomForestClassifier( n_estimators = 100 )
	# 创建十折交叉验证对象
	rf_classifier.fit( X_train, y_train )
	# 执行十折交叉验证
	scores = cross_val_score( rf_classifier, X_train, y_train, cv = kfold )
	# 输出每折的准确率
	for i, score in enumerate( scores ):
		print( "Fold {}: {:.4f}".format( i + 1, score ) )
	# 输出平均准确率
	print("随机森林")
	print( "Average Accuracy: {:.4f}".format( scores.mean( ) ) )
	predict_y = rf_classifier.predict( X_test )
	
	print( "mean_squared_error:", mean_squared_error( y_test, predict_y ) )
	print( 'accuracy_score：', accuracy_score( y_test, predict_y ) )



import time

start_time = time.time( )

from sklearn.preprocessing import LabelEncoder


nb_classes = 2


print( '读取数据' )
all = pd.read_csv( r"/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02汇总.csv" )
train = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Trainset.csv" )
test = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Testset.csv" )
PulFunction = ['FEV1/FVC%', 'FEV1-BEST', 'FVCpred', 'FEV1%Pred', 'FVC%Pred', 'RV/TLC%', 'DLCO%pred']
features = ['HTN', 'DM', 'CHD', 'Gender', 'smoking index', 'BMI', 'COPD course', 'Cl', 'hs-cTn', 'D-D', 'PH', 'OI', 'PCO2', 'NEU%', 'HGB', 'PLT', 'PASP']
# 加上心血管参数
# features = ['HTN', 'DM', 'CHD', 'Gender', 'smoking index', 'BMI', 'COPD course', 'Cl', 'hs-cTn', 'D-D', 'PH', 'OI', 'PCO2', 'NEU%', 'HGB', 'PLT', 'PASP', 'PAC', 'PVC', 'RBBB', 'SVT', 'AF',
# 'LBBB','HF']
X_train = train[features]
X_test = test[features]
X = pd.concat( [X_train, X_test], axis = 1 )
y_train = train["GOLD1-4"]
y_test = test["GOLD1-4"]

y = pd.concat( [y_train, y_test], axis = 1 )
y_train = LabelEncoder( ).fit_transform( y_train.values.ravel( ) )
y_test = LabelEncoder( ).fit_transform( y_test.values.ravel( ) )
y = LabelEncoder( ).fit_transform( y.values.ravel( ) )

# 训练集和测试集划分
# X_train, X_test, y_train, y_test = model_selection.train_test_split( x, y, test_size = 0.2, random_state = 1, shuffle = True )

# 搭建模型
clf = svm.SVC( kernel = 'rbf', gamma = 0.01,  # 核函数
               decision_function_shape = 'ovo',  # one vs one 分类问题
               C = 100 )

clf.fit( X_train, y_train )  # 训练
# y_train_hat=clf.predict(X_train)
# y_test_hat=clf.predict(X_test)
# 预测
train_predict = clf.predict( X_train )
test_predict = clf.predict( X_test )

# 准确率
train_acc = accuracy_score( y_train, train_predict )
test_acc = accuracy_score( y_test, test_predict )

# train_acc=clf.score(X_train, y_train)            #训练集的准确率
# test_acc=clf.score(X_test,y_test)                #测试集的准确率

print( "SVM训练集准确率: {0:.3f}, SVM测试集准确率: {1:.3f}".format( train_acc, test_acc ) )

## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix( test_predict, y_test )
np.set_printoptions( precision = 2 )
confusion_matrix = confusion_matrix_result.astype( 'float' ) / confusion_matrix_result.sum( axis = 1 )[:, np.newaxis]
plt.figure( figsize = (8, 6), dpi = 80 )
sns.heatmap( confusion_matrix, annot = True, cmap = 'Blues' )
plt.xlabel( 'Predicted labels' )
plt.ylabel( 'True labels' )
# print(confusion_matrix)
# plt.show( )

# K折交叉验证模块
from sklearn.model_selection import cross_val_score

# 使用K折交叉验证模块
scores = cross_val_score( clf, X_train, y_train, cv = 10, scoring = 'accuracy' )
# 将10次的预测准确率打印出
print( scores )
# [0.92 1.   0.83 0.88 0.91 0.96 1.   1.   0.78 0.74]
# 将10次的预测准确平均率打印出0.901630434782608
print( scores.mean( ) )

# 基于svm 实现分类  # 基于网格搜索获取最优模型
from sklearn.model_selection import GridSearchCV

model = svm.SVC( probability = True )
params = [
		{ 'kernel': ['linear'], 'C': [1, 10, 100, 1000] },
		{ 'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3] },
		{
				'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
				'gamma' : [1, 0.1, 0.01, 0.001]
				}]
model = GridSearchCV( estimator = model, param_grid = params, cv = 5 )
model.fit( X_train, y_train )

# 网格搜索训练后的副产品
print( "模型的最优参数：", model.best_params_ )
print( "最优模型分数：", model.best_score_ )
print( "最优模型对象：", model.best_estimator_ )
if __name__ == '__main__':
	train = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Trainset.csv" )
	test = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Testset.csv" )
	PulFunction = ['FEV1/FVC%', 'FEV1-BEST', 'FVCpred', 'FEV1%Pred', 'FVC%Pred', 'RV/TLC%', 'DLCO%pred']
	features= ['HTN', 'DM', 'CHD', 'Gender', 'smoking index', 'BMI', 'COPD course', 'Cl', 'hs-cTn', 'D-D', 'PH', 'OI', 'PCO2', 'NEU%', 'HGB', 'PLT', 'PASP'," 'FEV1-BEST"]
	#加上心血管参数
	features = ['HTN', 'DM', 'CHD', 'Gender', 'smoking index', 'BMI', 'COPD course', 'Cl', 'hs-cTn', 'D-D', 'PH', 'OI', 'PCO2', 'NEU%', 'HGB', 'PLT', 'PASP','PAC', 'PVC', 'RBBB', 'SVT', 'AF', 'LBBB', 'HF',
	            
	            'FEV1-BEST']
	X_train = train[features]
	X_test=test[features]

	y_train = train["GOLD1-4"]
	y_test = test["GOLD1-4"]
	y_train = LabelEncoder( ).fit_transform( y_train.values.ravel( ) )
	y_test= LabelEncoder( ).fit_transform( y_test.values.ravel( ) )
	modelclassfication(X_train,X_test,y_train,y_test,10)

# # 系数输出
# dic = { 'features': X_train.columns, '系数': lasso.coef_ }
# df0=pd.DataFrame(dic)
# df00=df0[df0['系数'] != 0]
# features=df00["features"].tolist()
# print(len(features),features)
#
#