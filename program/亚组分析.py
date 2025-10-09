from warnings import filterwarnings

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, matthews_corrcoef, f1_score, cohen_kappa_score
from sklearn import metrics
from matplotlib import pyplot
from numpy import argmax
from functools import reduce
import shap
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

#
pd.set_option( 'display.max_rows', None )
pd.set_option( 'display.width', None )
filterwarnings( "ignore" )
import math


def _centered ( arr, newsize ):
	# Return the center newsize portion of the array.
	newsize = np.asarray( newsize )
	currsize = np.array( arr.shape )
	startind = (currsize - newsize) // 2
	endind = startind + newsize
	myslice = [slice( startind[k], endind[k] ) for k in range( len( endind ) )]
	return arr[tuple( myslice )]


scipy.signal.signaltools._centered = _centered

features = ['Age', 'smoking index', 'FEV1/FVC%', 'FEV1%Pred', 'AST', 'NT-proBNP', 'hs-cTn',
            'OI', 'PCO2', 'CRP', 'HGB', 'PAHD']
features = ['Age', 'smoking index', 'FEV1/FVC%', 'FEV1%Pred', 'AST', 'NT-proBNP', 'hs-cTn', 'PCO2', 'CRP', 'HGB', 'PAHD']
datadf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02汇总.csv" )
traindf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Trainset.csv" )[features].dropna( )
testdf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/05亚组分析-COPD一二级.csv" )[features].dropna( )
testdf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/05亚组分析-COPD三四级.csv" )[features].dropna( )

# target = "PASP"
target = "PAHD"
# print( traindf.isnull( ).sum( ).sort_values( ascending = False ) )

# ===========================数据集划分===============================
from sklearn.utils import shuffle

traindf = shuffle( traindf )

mms = MinMaxScaler( )
led = LabelEncoder( )
num_cols = []
cate_cols = []

# 读取训练集
X_train = traindf.drop( [target], axis = 1 )
for col in X_train:
	if X_train[col].dtype == "float64":
		num_cols.append( col )
	if X_train[col].dtype == "object":
		cate_cols.append( col )

for col in cate_cols:
	X_train[col] = led.fit_transform( X_train[col] )
X_train[num_cols] = mms.fit_transform( X_train[num_cols] )
# y_train = traindf[target].values.reshape( -1, 1 )
# y_train = mms.fit_transform( y_train )
y_train = traindf[target]

y_train = led.fit_transform( y_train.values.ravel( ) )




#亚组分析
# 读取测试集

X_test = testdf.drop( [target], axis = 1 )
for col in X_test:
	if X_test[col].dtype == "float64":
		num_cols.append( col )
	if X_test[col].dtype == "object":
		cate_cols.append( col )

for col in cate_cols:
	X_test[col] = led.fit_transform( X_test[col] )
X_test[num_cols] = mms.fit_transform( X_test[num_cols] )

# y_test= testdf[target].values.reshape( -1, 1 )
# y_test = mms.fit_transform( y_test )
y_test = testdf[target]

y_test = led.fit_transform( y_test.values.ravel( ) )




# ===========绘制ROC曲线================
def Draw_ROC ( X_train, y_train, model ):
	list1 = model.predict( X_train )
	list2 = y_train
	fpr_model, tpr_model, thresholds = roc_curve( list1, list2, pos_label = 1 )
	roc_auc_model = auc( fpr_model, tpr_model )
	# font = { 'family': 'Times New Roman', 'size': 12, }
	# sns.set( font_scale = 1.2 )
	# plt.rc( 'font', family = 'Times New Roman' )
	# plt.plot( fpr_model, tpr_model, 'blue', label = 'AUC = %0.2f' % roc_auc_model )
	#
	# plt.legend( loc = 'lower right', fontsize = 12 )
	# plt.plot( [0, 1], [0, 1], 'r--' )
	# plt.xlim( [0.0, 1.0] )
	# plt.ylim( [0.0, 1.05] )
	# plt.ylabel( 'True  Positive Rate', fontsize = 14 )
	# plt.title( 'Receiver Operating Characteristi' )
	#
	# plt.xlabel( 'Flase  Positive Rate', fontsize = 14 )
	# plt.show( )
	# print(model, "auc", roc_auc_model )
	
	return fpr_model, tpr_model, roc_auc_model


# ===========绘制特征多样性===============

def ABS_SHAP ( df_shap, df ):
	# import matplotlib as plt
	# Make a copy of the input data
	shap_v = pd.DataFrame( df_shap )
	feature_list = df.columns
	shap_v.columns = feature_list
	df_v = df.copy( ).reset_index( ).drop( 'index', axis = 1 )
	
	# Determine the correlation in order to plot with different colors
	corr_list = list( )
	for i in feature_list:
		b = np.corrcoef( shap_v[i], df_v[i] )[1][0]
		corr_list.append( b )
	corr_df = pd.concat( [pd.Series( feature_list ), pd.Series( corr_list )], axis = 1 ).fillna( 0 )
	# Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
	corr_df.columns = ['Variable', 'Corr']
	corr_df['Sign'] = np.where( corr_df['Corr'] > 0, 'red', 'blue' )
	
	# Plot it
	shap_abs = np.abs( shap_v )
	k = pd.DataFrame( shap_abs.mean( ) ).reset_index( )
	k.columns = ['Variable', 'SHAP_abs']
	k2 = k.merge( corr_df, left_on = 'Variable', right_on = 'Variable', how = 'inner' )
	k2 = k2.sort_values( by = 'SHAP_abs', ascending = True )
	colorlist = k2['Sign']
	ax = k2.plot.barh( x = 'Variable', y = 'SHAP_abs', color = colorlist, figsize = (5, 6), legend = False )
	ax.set_xlabel( "SHAP Value (Red = Positive Impact)" )


def XGBmodel ( X_train, y_train, X_test, y_test, k ):
	print( "————XGBmodel——————" )
	model = XGBClassifier( ).fit( X_train, y_train )
	y_pred = model.predict( X_test )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	
	model.get_params( )  # 获取参数
	params = {
			"learning_rate"   : [0.1, 0.5],  # 学习率
			"n_estimators"    : [100, 500, 1000],  # 树的个数
			"max_depth"       : [5, 8],  # 树的最大深度
			"min_child_weight": [3, 4, 5],  # 叶子节点样本权重加和最小值sum(H)
			"gamma"           : [0],  # 节点分裂所需的最小损失函数下降值
			"subsample"       : [0.4, 0.6, 0.8],  # 样本随机采样作为训练集的比例
			"colsample_bytree": [0.4, 0.6, 0.8],  # 使用特征比例
			"objective"       : ['multi:softmax'],  # 损失函数(这里为多分类）
			"num_class"       : [2],  # 多分类问题类别数
			"seed"            : [1]
			}
	# 网格搜索 自动调参
	# model_cv = GridSearchCV( model, params, cv = k, n_jobs = -1, verbose = 2 ).fit( X_train, y_train )
	# 	# print( model_cv.best_score_ )
	# 	# print( model_cv.best_params_ )
	
	#
	# 测试集
	model_tuned = XGBClassifier( colsample_bytree = 0.4,
	                             gamma = 0,
	                             learning_rate = 0.1,
	                             max_depth = 5,
	                             min_child_weight = 4,
	                             n_estimators = 100,
	                             num_class = 2,
	                             objective = 'multi:softmax',
	                             seed = 1,
	                             subsample = 0.6 ).fit( X_train, y_train )
	
	y_pred = model_tuned.predict( X_test )
	
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	return y_test, y_pred, model_tuned


def rfmodel ( X_train, y_train, X_test, y_test, k ):
	print( "随机森林" )
	model = RandomForestClassifier( ).fit( X_train, y_train )
	y_pred = model.predict( X_test )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	
	model.get_params( )  # 获取参数
	rf_params = {
			"max_depth"        : [2, 3, 5, 8, 10],
			"max_features"     : [2, 5, 8],
			"n_estimators"     : [10, 100, 500, 1000],
			"min_samples_split": [2, 5, 10]
			}
	# 自动调参
	# rf_cv = GridSearchCV( model, rf_params, cv = k, n_jobs = -1, verbose = 2 )
	# rf_cv.fit( X_train, y_train )
	# print( "Best Score: " + str( rf_cv.best_score_ ) )
	# print( "Best Parameters: " + str( rf_cv.best_params_ ) )
	
	rf_tuned = RandomForestClassifier( max_depth = 3, max_features = 8, min_samples_split = 10, n_estimators = 100 )
	rf_tuned.fit( X_train, y_train )
	y_pred = rf_tuned.predict( X_test )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	return y_test, y_pred, rf_tuned
	
	''''
	# # 特征重要性评价整理为数据框
	# Importance = pd.DataFrame( { "Importance": rf_tuned.feature_importances_ * 100 },
	#                            index = X_train.columns )
	# # 降序排列并画柱状图
	# Importance.sort_values( by = "Importance",
	#                         axis = 0,
	#                         ascending = True ).plot( kind = "barh", color = "darkorange" )
	# plt.xlabel( "Variables Importance Ratio" )
	# plt.show()
	'''


def mlpmodel ( X_train, y_train, X_test, y_test, k ):
	print( "多层感知机" )
	mlpc = MLPClassifier( ).fit( X_train, y_train )
	# y_pred = mlpc.predict( X_test )
	# print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	#
	# mlpc_params = {
	# 		"alpha"             : [0.1, 0.01, 0.02, 0.005, 0.0001, 0.00001],  # 默认0.0001,正则化项参数
	# 		"hidden_layer_sizes": [(10, 10, 10),  # 隐藏神经元
	# 		                       (100, 100, 100),
	# 		                       (100, 100),
	# 		                       (3, 5),
	# 		                       (5, 3)],
	# 		"solver"            : ["lbfgs", "adam", "sgd"],  # 默认adam，用来优化权重
	# 		"activation" : ['identity','logistic','tanh','relu']
	# 		}
	# mlpc = MLPClassifier( )
	# mlpc_cv = GridSearchCV( mlpc, mlpc_params, cv = k, n_jobs = -1, verbose = 2 )  # 自动调参
	# mlpc_cv.fit( X_train, y_train )
	# 最佳参数
	# print( "Best Score: " + str( mlpc_cv.best_score_ ) )
	# print( "Best Parameters: " + str( mlpc_cv.best_params_ ) )
	mlpc_tuned = MLPClassifier( activation = "tanh", alpha = 0.1, hidden_layer_sizes = (10, 10, 10), solver = "adam" )  # 利用最佳参数建模
	mlpc_tuned.fit( X_train, y_train )
	y_pred = mlpc_tuned.predict( X_test )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	
	return y_test, y_pred, mlpc_tuned


def GBMmodel ( X_train, y_train, X_test, y_test, k ):
	print( "GBM (Gradient Boosting Machine)" )
	gbm_model = GradientBoostingClassifier( ).fit( X_train, y_train )
	y_pred = gbm_model.predict( X_test )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	
	gbm_model.get_params( )
	gbm_params = {
			"learning_rate"    : [0.001, 0.01, 0.1, 0.05],
			"n_estimators"     : [100, 500, 1000],
			"max_depth"        : [3, 5, 10],
			"min_samples_split": [2, 5, 10]
			}
	
	# gbm = GradientBoostingClassifier( )
	# gbm_cv = GridSearchCV( gbm, gbm_params, cv = k, n_jobs = -1, verbose = 2 )
	# gbm_cv.fit( X_train, y_train )
	# print( "Best Score: " + str( gbm_cv.best_score_ ) )
	# print( "Best Parameters: " + str( gbm_cv.best_params_ ) )
	# 通过最佳参数建模
	gbm_tuned = GradientBoostingClassifier( learning_rate = 0.001,
	                                        max_depth = 3,
	                                        min_samples_split = 2,
	                                        n_estimators = 1000 ).fit( X_train, y_train )
	y_pred = gbm_tuned.predict( X_test )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	
	return y_test, y_pred, gbm_tuned


def KNNmodel ( X_train, y_train, X_test, y_test, k ):
	print( "KNN Model" )
	knn = KNeighborsClassifier( )
	knn_model = knn.fit( X_train, y_train )
	y_pred = knn_model.predict( X_test )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	
	# knn_params = { "n_neighbors": np.arange( 1, 50 ) }
	# knn = KNeighborsClassifier( )
	# knn_cv = GridSearchCV( knn, knn_params, cv = k )
	# knn_cv.fit( X_train, y_train )
	# # 模型优化
	# print( "Best Score: " + str( knn_cv.best_score_ ) )
	# print( "Best Parameters: " + str( knn_cv.best_params_ ) )
	
	#
	knn = KNeighborsClassifier( 25 )
	knn_tuned = knn.fit( X_train, y_train )
	knn_tuned.score( X_test, y_test )
	y_pred = knn_tuned.predict( X_test )
	scores = accuracy_score( y_test, y_pred )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	
	return y_test, y_pred, knn_tuned


def DessionTreemodel ( X_train, y_train, X_test, y_test, k ):
	print( "决策树" )
	cart = DecisionTreeClassifier( )
	cart_model = cart.fit( X_train, y_train )
	y_pred = cart_model.predict( X_test )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	
	# cart_params = {
	# 		"max_depth"        : list( range( 1, 10 ) ),
	# 		"min_samples_split": list( range( 2, 50 ) )
	# 		}
	# cart = tree.DecisionTreeClassifier( )
	# cart_cv = GridSearchCV( cart, cart_params, cv = k, n_jobs = -1, verbose = 2 )
	# cart_cv_model = cart_cv.fit( X_train, y_train )
	# print( "Best Score: " + str( cart_cv_model.best_score_ ) )
	# print( "Best Parameters: " + str( cart_cv_model.best_params_ ) )
	
	cart_tuned = tree.DecisionTreeClassifier( max_depth = 9, min_samples_split = 3 ).fit( X_train, y_train )
	y_pred = cart_tuned.predict( X_test )
	accuracy_score( y_test, y_pred )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	
	return y_test, y_pred, cart_tuned


def GaussianNBmodel ( X_train, y_train, X_test, y_test, k ):
	print( '朴素贝叶斯GaussianNB' )
	nb_model = GaussianNB( )
	nb_model.fit( X_train, y_train )
	y_pred = nb_model.predict( X_test )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	
	# print(nb_model.get_params( ))
	# params={ 'priors': [(0, 3296089385, 0, 6703910615 ),(0.55,0.45),(0.2,0.8),(0.8,0.2),(0.1,0.9),(0.05,0.095)], 'var_smoothing': [0.1, 0.01, 0.02, 1,2,5,10,100] }
	# #
	# # # # 网格搜索 自动调参
	# model_cv = GridSearchCV( nb_model, params, cv = k, n_jobs = -1, verbose = 2 ).fit( X_train, y_train )
	# print( model_cv.best_score_ )
	# print( model_cv.best_params_ )
	nb_model = GaussianNB( priors = (0.1, 0.9), var_smoothing = 0.1 )
	nb_model = nb_model.fit( X_train, y_train )
	y_pred = nb_model.predict( X_test )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	return y_test, y_pred, nb_model


def drawTree ( model, features, labels ):
	with open( "tree.dot", 'w' ) as f:
		f = tree.export_graphviz( model,
		                          out_file = f,
		                          feature_names = features,
		                          class_names = labels )
	with(open( "tree.dot" )) as f:
		dot_graph = f.read( )
	graph = graphviz.Source( dot_graph )
	graph.view( )


def CARTDecisionTree ( X_train, y_train, X_test, y_test, k ):
	print( "CART 决策树" )
	model = tree.DecisionTreeClassifier( )
	model.fit( X_train, y_train )
	y_pred = model.predict( X_test )
	print( model.get_params( ) )
	
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	params = {
			'ccp_alpha'        : [0.01, 0.1, 0.05], 'class_weight': ["balanced"], 'criterion': ['gini', 'entrop'],
			"max_depth"        : list( range( 1, 5 ) ),
			"min_samples_split": list( range( 1, 5 ) ),
			"min_samples_leaf" : list( range( 1, 5 ) ),
			"max_features"     : list( range( 1, 5 ) )
			}
	model = tree.DecisionTreeClassifier( )
	#
	# model_cv = GridSearchCV( model, params, cv = k ).fit( X_train, y_train )
	# print( model_cv.best_score_ )
	# print( model_cv.best_params_ )
	cart_turned = tree.DecisionTreeClassifier(
			ccp_alpha = 0.01,
			class_weight = 'balanced',
			criterion = 'gini',  # 采用gini还是entropy进行特征选择
			max_depth = 4,  # 树的最大深度
			min_samples_split = 2,  # 内部节点分裂所需要的最小样本数量
			min_samples_leaf = 1,  # 叶子节点所需要的最小样本数量
			max_features = 2  # 寻找最优分割点时的最大特征数
			).fit( X_train, y_train )
	# # 对模型进行训练
	y_pred = cart_turned.predict( X_test )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	features = X_train.columns
	label = ["PAHD", "Without PAHD"]
	# drawTree( cart_turned, features, label )
	return y_test, y_pred, cart_turned


def Adamodel ( X_train, y_train, X_test, y_test, k ):
	print( "AdaBoost" )
	ada = AdaBoostClassifier( )
	ada.fit( X_train, y_train )
	y_pred = ada.predict( X_test )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( ada.get_params( ) )
	params = { 'learning_rate': [0.01, 0.1, 0.05, 0.5, 1], 'n_estimators': [50, 100, 200, 500], 'random_state': [0, 3, 6, 9] }
	#
	# model_cv = GridSearchCV( model, params, cv = k ).fit( X_train, y_train )
	# print( model_cv.best_score_ )
	# print( model_cv.best_params_ )
	
	ada_tuned = AdaBoostClassifier( learning_rate = 0.1, n_estimators = 50, random_state = 0 ).fit( X_train, y_train )
	y_pred = ada_tuned.predict( X_test )
	print( "mean_squared_error:", mean_squared_error( y_test, y_pred ) )
	print( 'accuracy_score：', accuracy_score( y_test, y_pred ) )
	print( classification_report( y_test, y_pred ) )
	
	return y_test, y_pred, ada_tuned


def find_optimal_cutoff ( tpr, fpr, threshold ):
	optimal_idx = np.argmax( tpr - fpr )
	optimal_threshold = threshold[optimal_idx]
	return optimal_threshold


def best_confusion_matrix ( y_test, y_test_predprob ):
	"""
        根据真实值和预测值（预测概率）的向量来计算混淆矩阵和最优的划分阈值

        Args:
            y_test:真实值
            y_test_predprob：预测值

        Returns:
            返回最佳划分阈值和混淆矩阵
        """
	
	fpr, tpr, thresholds = roc_curve( y_test, y_test_predprob, pos_label = 1 )
	cutoff = find_optimal_cutoff( tpr, fpr, thresholds )
	y_pred = list( map( lambda x: 1 if x >= cutoff else 0, y_test_predprob ) )
	TN, FP, FN, TP = confusion_matrix( y_test, y_pred ).ravel( )
	return cutoff, TN, FN, FP, TP


def evaluation ( clf, X_train, y_train, X_test, y_test, modelname, digits ):
	"""
		计算各个模型评价指标

		Args:
			clf：已经fit好的模型
			X_train,y_train,X_test,y_test:	训练和测试数据集
			modelname：模型名称，为了表格的绘制
			digits：各个评价指标需要保留的位数
		Returns:
			返回单个模型评价指标表格
		"""
	
	y_train_predprob = clf.predict_proba( X_train )[:, 1]
	train_auc = round( roc_auc_score( y_train, y_train_predprob ), digits )
	
	y_test_predprob = clf.predict_proba( X_test )[:, 1]
	test_auc = round( roc_auc_score( y_test, y_test_predprob ), digits )
	
	train_cutoff, TN1, FN1, FP1, TP1 = best_confusion_matrix( y_train, y_train_predprob )
	test_cutoff, TN2, FN2, FP2, TP2 = best_confusion_matrix( y_test, y_test_predprob )
	
	# Sen Spe
	best_recall, best_prec = round( TP2 / (TP2 + FN2), digits ), round( TN2 / (FP2 + TN2), digits )
	
	# PPV NPV
	npv, ppv = round( TN2 / (FN2 + TN2), digits ), round( TP2 / (TP2 + FP2), digits )
	
	# PLR NLR
	plr, nlr = round( (TP2 / (TP2 + FN2)) / (FP2 / (FP2 + TN2)), digits ), round( (FN2 / (TP2 + FN2)) / (TN2 / (FP2 + TN2)), digits )
	
	# F1值
	y_test_pred = list( map( lambda x: 1 if x >= test_cutoff else 0, y_test_predprob ) )
	f1 = round( f1_score( y_test, y_test_pred ), digits )
	
	# Youden Index
	youden = round( TP2 / (TP2 + FN2) + TN2 / (FP2 + TN2) - 1, digits )
	
	# MCC
	mcc = round( matthews_corrcoef( y_test, y_test_pred ), digits )
	
	# Kappa
	kappa = round( cohen_kappa_score( y_test_pred, y_test ), digits )
	Model = ['train_auc', 'test_auc', 'specificity', 'sensitivity', 'F1', 'Youden Index', 'MCC', 'Kappa', 'npv', 'ppv', 'plr', 'nlr']
	
	Names = [train_auc, test_auc, best_prec, best_recall, f1, youden, mcc, kappa, npv, ppv, plr, nlr]
	dict = { }
	
	for i, j in enumerate( Model ):
		dict[j] = Names[i]
	return dict, modelname


def score ( ls_right_value, ls_left_value, ls_or, ls_xvar ):  # 提供列线图右边的数值和左边的数值,分类变量为1和0,多分类变量为多个1和0
	ls_beta = [np.log( x ) for x in ls_or]
	ls_beta_abs = [np.abs( x ) for x in ls_beta]
	ls_distance_abs = [np.abs( a - b ) for a, b in zip( ls_right_value, ls_left_value )]  # 各自标尺的右边数值与左边数值的差
	ls_pi_pre = [a * b for a, b in zip( ls_beta_abs, ls_distance_abs )]
	ls_max_score = []  # 求各个变量最大的得分
	for pi_pre in ls_pi_pre:
		max_score = np.divide( pi_pre, np.max( ls_pi_pre ) ) * 100
		ls_max_score.append( max_score )
	ls_unit_score = [a / b for a, b in zip( ls_max_score, ls_distance_abs )]  # 求各个变量每个刻度单位的得分
	ls_actual_distance = [a - b for a, b in zip( ls_xvar, ls_left_value )]  # 求实际的总得分
	ls_actual_distance_abs = map( np.abs, ls_actual_distance )
	ls_score = [a * b for a, b in zip( ls_unit_score, ls_actual_distance_abs )]
	total_score = 0
	for i, val in enumerate( ls_score ):
		total_score += ls_score[i]
	return ls_score, total_score


if __name__ == "__main__":
	k = 10
	y_test, y_pred, rf = rfmodel( X_train, y_train, X_test, y_test, k )
	# Draw_ROC(y_test,y_pred)
	y_test, y_pred,xgb = XGBmodel( X_train, y_train, X_test, y_test, k )
	# Draw_ROC( y_test, y_pred )
	y_test, y_pred,mlp = mlpmodel( X_train, y_train, X_test, y_test, k )
	# Draw_ROC( y_test, y_pred )
	y_test, y_pred,gbm = GBMmodel( X_train, y_train, X_test, y_test, k )
	# Draw_ROC( y_test, y_pred )
	y_test, y_pred,knn = KNNmodel( X_train, y_train, X_test, y_test, k )
	# Draw_ROC( y_test, y_pred )
	y_test, y_pred,dt = DessionTreemodel( X_train, y_train, X_test, y_test, k )
	# Draw_ROC( y_test, y_pred )
	y_test, y_pred,gnb = GaussianNBmodel( X_train, y_train, X_test, y_test, k )
	# Draw_ROC( y_test, y_pred )
	y_test, y_pred,cart = CARTDecisionTree( X_train, y_train, X_test, y_test, k )
	# Draw_ROC( y_test, y_pred )
	y_test, y_pred, ada = Adamodel( X_train, y_train, X_test, y_test, k )
	
	# 绘制所有roc在一张图
	# Draw_ROC( y_test, y_pred )
	models = [rf,xgb,mlp,gbm,knn,dt,gnb,cart,ada]
	model_name=["RF","XGBoost","MLP","GBM","KNN","DescionTree","GaussianNB","CART","AdaBoost"]

	fprs=[]
	tprs=[]
	aucs=[]
	for i in models:
		fpr_model, tpr_model, roc_auc_model=Draw_ROC(X_train,y_train,i)
		fprs.append(fpr_model)
		tprs.append(tpr_model)
		aucs.append(roc_auc_model)
	#
	# plt.rcParams['font.family'] = ['Times New Roman']
	# plt.rcParams['figure.figsize'] = (10,10)
	# for i in range(len(fprs)):
	# 	plt.plot(fprs[i],tprs[i],label=model_name[i]+"_AUC=%0.3f" %aucs[i])
	# plt.title( 'Receiver Operating Characteristic' )
	# plt.legend( loc = 'lower right' )
	# plt.plot( [0, 1], [0, 1], 'r--' )
	# plt.xlim( [-0.05, 1.05] )
	# plt.ylim( [-0.05, 1.05] )
	# plt.ylabel( 'True positive Rate' )
	# plt.xlabel( 'False positive Rate' )
	# plt.grid( linestyle = '-.' )
	# plt.grid( True )
	# plt.show( )
	df=pd.DataFrame()

	for i in range(len(fprs)):
		info,name=evaluation(models[i],X_train,y_train,X_test,y_test,model_name[i],4)
		df[name]=info
	print(df)
	df.to_csv("/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/06亚组分析结果.csv",header=True,index=True)
	# models = [rf,xgb,mlp,gbm,knn,dt,gnb,cart,ada]
	models = [rf]
	for model in models:
		
		shap_values = shap.TreeExplainer( model ).shap_values( X_test )
		shap.summary_plot( shap_values, X_train, plot_type = "bar" )
		ABS_SHAP( shap_values, X_train )
		
		shap.summary_plot( shap_values, X_train )
		plt.show( )