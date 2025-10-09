import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from statsmodels.discrete.discrete_model import MNLogit
features = ['Age', 'smoking index', 'FEV1/FVC%', 'FEV1%Pred', 'AST', 'NT-proBNP', 'hs-cTn',
            'OI', 'PCO2', 'CRP', 'HGB', 'PAHD']
datadf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02汇总.csv" )
traindf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Trainset.csv" )[features].dropna( )
testdf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Testset.csv" )[features].dropna( )
target = "PASP"
target = "PAHD"
print( datadf.columns )
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

from sklearn.linear_model import LogisticRegression
model= MNLogit
model_LR = MNLogit( y_train, X_train ).fit( )

print( model_LR.summary( ) )


# Precision Recall 和 F1
nb_classes = 2  # 设置分类数
y_pred = model_LR.predict( X_test )

y_pred_max = [np.argmax( y ) for y in y_pred]  # 取出y中元素最大值所对应的索引
y_true = y_train

y_pred_b = label_binarize( y_pred_max, classes = [i for i in range( nb_classes )] )  # binarize the Output
y_true_b = label_binarize( y_true, classes = [i for i in range( nb_classes )] )  # binarize the Output

precision = precision_score( y_true_b, y_pred_b, average = 'micro' )
recall = recall_score( y_true_b, y_pred_b, average = 'micro' )
f1_score = f1_score( y_true_b, y_pred_b, average = 'micro' )

print( "Precision_score:", precision )
print( "Recall_score:", recall )
print( "F1_score:", f1_score )
import matplotlib.pyplot as plt

Y_valid = y_true_b

Y_pred = model_LR.predict( )  # 用预测的概率值，不然ROC是折线的形式

fpr = dict( )
tpr = dict( )
roc_auc = dict( )
for i in range( nb_classes ):
	fpr[i], tpr[i], _ = roc_curve( Y_valid[:, i], Y_pred[:, i] )
	roc_auc[i] = auc( fpr[i], tpr[i] )

fpr["micro"], tpr["micro"], _ = roc_curve( Y_valid.ravel( ), Y_pred.ravel( ) )
roc_auc["micro"] = auc( fpr["micro"], tpr["micro"] )

lw = 2
plt.figure( )
plt.plot( fpr["micro"], tpr["micro"],
          label = 'micro-average ROC curve (area = {0:0.2f})'
                  ''.format( roc_auc["micro"] ),
          color = 'deeppink', linestyle = ':', linewidth = 4 )

colors = cycle( ['darkorange', 'cornflowerblue'] )  # 几个类别就设置几个颜色
for i, color in zip( range( nb_classes ), colors ):
	plt.plot( fpr[i], tpr[i], color = color, lw = lw,
	          label = 'ROC curve of class {0} (area = {1:0.2f})'
	                  ''.format( i, roc_auc[i] ) )

plt.plot( [0, 1], [0, 1], 'k--', lw = lw )  # 斜着的分界线
plt.xlim( [0.0, 1.0] )
plt.ylim( [0.0, 1.05] )
plt.xlabel( 'FPR' )
plt.ylabel( 'TPR' )
plt.title( 'ROC' )
plt.legend( loc = "lower right" )
plt.show( )
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.family']='sans-serif'  # 解决绘图中文乱码
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为框

# 混淆矩阵
classes = ['COPD-1', 'COPD-2', 'COPD-3', 'COPD-4']

confusion_matrix = model_LR.pred_table( )

plt.figure( figsize = (6, 4), dpi = 90 )

plt.imshow( confusion_matrix, interpolation = 'nearest', cmap = plt.cm.Oranges )  # 按照像素显示出矩阵
plt.title( 'confusion_matrix' )
plt.colorbar( )

tick_marks = np.arange( 4 )
plt.xticks( tick_marks, classes )
plt.yticks( tick_marks, classes, rotation = 90, verticalalignment = 'center' )

for x in range( len( confusion_matrix ) ):
	for y in range( len( confusion_matrix ) ):
		plt.annotate( confusion_matrix[y, x], xy = (x, y), horizontalalignment = 'center', verticalalignment = 'center' )

plt.ylabel( 'Ground Truth' )
plt.xlabel( 'Prediction' )
plt.show( )