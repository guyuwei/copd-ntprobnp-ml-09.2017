'''
Lasso变量选择，某些自变量系数估计值压缩为零。
首先构建包含所有自变量的的模型，注意要对属性类型进行因子化，即转换成哑变量。
'''
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split, LeaveOneOut, GridSearchCV, permutation_test_score, cross_val_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.stats import pearsonr, ttest_ind, levene
import itertools
import time

import numpy as np
import warnings

warnings.filterwarnings( action = 'ignore' )
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
import seaborn as sns
pd.set_option( "display.max_rows", None )
pd.set_option( "display.max_columns", None )
pd.set_option( "display.expand_frame_repr", False )

from sklearn.preprocessing import MinMaxScaler

features = ['Age', 'Gender', 'smoking index', 'BMI', 'COPD course', 'FEV1/FVC%',
 'FEV1%Pred', 'ALT', 'AST', 'Cr', 'K', 'Na', 'Cl', 'NT-proBNP', 'hs-cTn',
 'D-D', 'PH', 'OI', 'PCO2', 'CRP', 'WBC', 'NEU%', 'EOS%', 'HGB', 'HTN',
 'DM', 'CHD', 'COPD&PN', 'PAC', 'PVC', 'RBBB', 'AF', 'LBBB',
 'SVT', 'HF', 'PAHD', 'GOLD1-4']
datadf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02汇总.csv" )
traindf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Trainset.csv" )[features].dropna()
testdf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Testset.csv" )[features].dropna()
target="PASP"
target="PAHD"
print( datadf.columns )
# print( traindf.isnull( ).sum( ).sort_values( ascending = False ) )

# ===========================数据集划分===============================
from sklearn.utils import shuffle
traindf = shuffle( traindf )

mms = MinMaxScaler( )
led = LabelEncoder( )
num_cols = []
cate_cols = []

#读取训练集
X_train = traindf.drop( [target], axis = 1 )
for col in X_train:
	if X_train[col].dtype == "float64":
		num_cols.append( col )
	if X_train[col].dtype == "object":
		cate_cols.append( col )

for col in cate_cols:
	X_train[col] = led.fit_transform( X_train[col] )
# X_train[num_cols] = mms.fit_transform( X_train[num_cols] )

# y_train = traindf[target].values.reshape( -1, 1 )
# y_train = mms.fit_transform( y_train )
y_train = traindf[target]

y_train = led.fit_transform( y_train.values.ravel( ) )

#读取测试集
X_test = testdf.drop( [target], axis = 1 )
for col in X_test:
	if X_test[col].dtype == "float64":
		num_cols.append( col )
	if X_test[col].dtype == "object":
		cate_cols.append( col )

for col in cate_cols:
	X_test[col] = led.fit_transform( X_test[col] )
# X_test[num_cols] = mms.fit_transform( X_test[num_cols] )

# y_test= testdf[target].values.reshape( -1, 1 )
# y_test = mms.fit_transform( y_test )
y_test= testdf[target]

y_test = led.fit_transform( y_test.values.ravel( ) )


# # ==========================训练集：Lasso变量筛选==========================
lasso_cofficients = []
# 构造空列表，用于存储模型的偏回归系数

alphas = np.logspace( -3, 1, 50 )
asso_cofficients = []
for Lambda in alphas:
	lasso_model = Lasso( alpha = Lambda, max_iter = 10000 )
	lasso_model.fit( X_train, y_train )
	lasso_cofficients.append( lasso_model.coef_ )





# =================================读取数据============================
class Solution( ):
	def __init__ ( self ):
	
		self.feature = features

	# =======================Lasso变量筛===============
	def optimal_lambda_value ( self ):
		Lambdas = np.logspace( -5, 2, 200 )  # 10的-5到10的2次方
		# 构造空列表，用于存储模型的偏回归系数
		lasso_cofficients = []
		for Lambda in Lambdas:
			lasso = Lasso( alpha = Lambda, max_iter = 10000 )
			lasso.fit( train_dataset, train_labels )
			lasso_cofficients.append( lasso.coef_ )
		# 绘制Lambda与回归系数的关系
		plt.plot( Lambdas, lasso_cofficients )
		# 对x轴作对数变换
		plt.xscale( 'log' )
		# 设置折线图x轴和y轴标签
		plt.xlabel( 'Lambda' )
		plt.ylabel( 'Cofficients' )
		# 显示图形
		plt.show( )
		# LASSO回归模型的交叉验证
		lasso_cv = LassoCV( alphas = Lambdas, cv = 10, max_iter = 10000 )
		lasso_cv.fit( train_dataset, train_labels )
		# 输出最佳的lambda值
		lasso_best_alpha = lasso_cv.alpha_
		print( lasso_best_alpha )
		return lasso_best_alpha
	
	# 基于最佳的lambda值建模
	def model ( self, train_dataset, train_labels, lasso_best_alpha ):
		lasso = Lasso( alpha = lasso_best_alpha,  max_iter = 10000 )
		lasso.fit( train_dataset, train_labels )
		return lasso
	
	def feature_importance ( self, lasso ):
		# 返回LASSO回归的系数
		dic = { '特征': train_dataset.columns, '系数': lasso.coef_ }
		df = pd.DataFrame( dic )
		df1 = df[df['系数'] != 0]
		print( df1 )
		print(df1["特征"].values)
		coef = pd.Series( lasso.coef_, index = train_dataset.columns )
		imp_coef = pd.concat( [coef.sort_values( ).head( 10 ), coef.sort_values( ).tail( 10 )] )
		sns.set( font_scale = 1.2 )
		# plt.rc('font', family='Times New Roman')
		plt.rc( 'font', family = 'simsun' )
		imp_coef.plot( kind = "barh" )
		plt.title( "Lasso回归模型" )
		plt.show( )
		return df1
	
	def prediction ( self, lasso ):
		# lasso_predict = lasso.predict(test_dataset)
		lasso_predict = np.round( lasso.predict( test_dataset ) )
		print( "测试集结果：", len(y_test),sum( lasso_predict == test_labels ) )
		print( metrics.classification_report( test_labels, lasso_predict ) )
		print( metrics.confusion_matrix( test_labels, lasso_predict ) )
		RMSE = np.sqrt( mean_squared_error( test_labels, lasso_predict ) )
		print( "RMSE",RMSE )
		return RMSE


if __name__ == "__main__":
	Object1 = Solution( )
	train_dataset, train_labels, test_dataset, test_labels = X_train,y_train,X_test,y_test
	lasso_best_alpha = Object1.optimal_lambda_value( )
	lasso = Object1.model( train_dataset, train_labels, lasso_best_alpha )
	feature_choose = Object1.feature_importance( lasso )
	print(feature_choose)
	RMSE = Object1.prediction( lasso )
	
	
	
	# verision 2
	X=X_train
	y=y_train
	import pandas as pd
	import sklearn
	from sklearn.utils import shuffle
	from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, mean_squared_error, r2_score
	from sklearn.model_selection import RepeatedKFold, train_test_split, LeaveOneOut, GridSearchCV, permutation_test_score, cross_val_score
	from sklearn import svm
	from sklearn.preprocessing import StandardScaler
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.pyplot import MultipleLocator
	from scipy.stats import pearsonr, ttest_ind, levene
	import itertools
	import time
	import matplotlib.pyplot as plt
	import matplotlib.ticker as ticker
	
	alphas = np.logspace( -3, 1, 50 )
	model_lassoCV = LassoCV( alphas = alphas, cv = 10, max_iter = 100000 ).fit( X, y )
	# model_lassoCV.predict(X)
	MSEs = model_lassoCV.mse_path_
	'''
	MSEs_mean, MSE_std = [],[]
	for i in range(len(MESs)):
	    MSEs_mean.append(MSEs[i].mean())
	    MSEs_std.append(MSEs[i].std())
	'''
	
	MSEs_mean = np.apply_along_axis( np.mean, 1, MSEs )
	MSEs_std = np.apply_along_axis( np.std, 1, MSEs )
	# print(np.mean(MSEs_std),np.mean(MSEs_mean))
	plt.figure( dpi = 200,figsize = (10,8) )  # dpi = 300
	plt.errorbar( model_lassoCV.alphas_, MSEs_mean
	              , yerr = MSEs_std
	              , fmt = 'o'
	              , ms = 5  # dot size
	              , mfc = 'r'  # dot color
	              , mec = 'r'  # dot margin color
	              , ecolor = 'grey'
	              , elinewidth =1.5  # error bar width
	              , capsize = 3  # cap length of error bar
	              , capthick = 1 )
	plt.semilogx( )
	plt.axvline( model_lassoCV.alpha_, color = 'black', ls = '--', )
	plt.xlabel( 'Lamda' )
	plt.ylabel( 'MSE' )
	ax = plt.gca( )
	y_major_locator = ticker.MultipleLocator( 0.05 )
	ax.yaxis.set_major_locator( y_major_locator )
	plt.show( )
	coef = pd.Series( model_lassoCV.coef_, index = X.columns )
	print( "Lasso picked " + str( sum( coef != 0 ) ) + " variables and eliminated the other " + str( sum( coef == 0 ) ) + ' variables' )
	index = coef[coef != 0].index
	X = X[index]
	print( coef[coef != 0] )
	
	#version3
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib
	from sklearn import model_selection
	from sklearn.linear_model import Lasso, LassoCV
	
	font = {
			'family': 'FangSong',
			'weight': 'bold',
			'size'  : 12
			}
	matplotlib.rc( "font", **font )

	# 构造不同的Lambda值
	Lambdas = np.logspace( -5, 2, 200 )
	# 构造空列表，用于存储模型的偏回归系数
	lasso_cofficients = []
	for Lambda in Lambdas:
		lasso = Lasso( alpha = Lambda,  max_iter = 10000 )
		lasso.fit( X_train, y_train )
		lasso_cofficients.append( lasso.coef_ )
	'''
	可视化方法确定λ的值
	'''
	# 绘制Lambda与回归线的折线图
	plt.plot( Lambdas, lasso_cofficients )
	# 对x轴做对数变换
	plt.xscale( 'log' )
	# 设置折线图x轴和y轴标签
	plt.xlabel( 'Lambda' )
	plt.ylabel( 'Cofficients' )
	# 显示图形
	plt.show( )
	# LASSO回归模型的交叉验证
	lasso_cv = LassoCV( alphas = Lambdas,  cv = 10, max_iter = 10000 )
	lasso_cv.fit( X_train, y_train )
	# 输出最佳的lambda值
	lasso_best_alpha = lasso_cv.alpha_  # 0.06294988990221888
	print( "____________VERSIon3:",lasso_best_alpha )
	
	# 基于最佳的lambda值建模
	lasso = Lasso( alpha = lasso_best_alpha, max_iter = 10000 )
	# 对"类"加以数据实体，执行回归系数的运算
	lasso.fit( X_train, y_train )
	# 返回LASSO回归的系数
	res = pd.Series( index = ['Intercept'] + X_train.columns.tolist( ), data = [lasso.intercept_] + lasso.coef_.tolist( ) )

	print( res )
	# 模型预测
	lasso_predict = lasso.predict( X_test )
	# 验证预测效果
	from sklearn.metrics import mean_squared_error
	
	RMSE = np.sqrt( mean_squared_error( y_test, lasso_predict ) )  # 53.061437258225745
	print( "version3 RMSE:",RMSE )