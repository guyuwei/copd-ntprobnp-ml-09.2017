import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
import  numpy as np
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


# coding: UTF-8
from __future__ import division
import numpy as np
import scipy as sp
from weakclassify import WEAKC
from dml.tool import sign


class ADABC:
	def __init__ ( self, X, y, Weaker = WEAKC ):
		'''
			Weaker is a class of weak classifier
			It should have a 	train(self.W) method pass the weight parameter to train
								pred(test_set) method which return y formed by 1 or -1
			see detail in <统计学习方法>
		'''
		self.X = np.array( X )
		self.y = np.array( y )
		self.Weaker = Weaker
		self.sums = np.zeros( self.y.shape )
		self.W = np.ones( (self.X.shape[1], 1) ).flatten( 1 ) / self.X.shape[1]
		self.Q = 0
	
	# print self.W
	def train ( self, M = 4 ):
		'''
			M is the maximal Weaker classification
		'''
		self.G = { }
		self.alpha = { }
		for i in range( M ):
			self.G.setdefault( i )
			self.alpha.setdefault( i )
		for i in range( M ):
			self.G[i] = self.Weaker( self.X, self.y )
			e = self.G[i].train( self.W )
			# print self.G[i].t_val,self.G[i].t_b,e
			self.alpha[i] = 1 / 2 * np.log( (1 - e) / e )
			# print self.alpha[i]
			sg = self.G[i].pred( self.X )
			Z = self.W * np.exp( -self.alpha[i] * self.y * sg.transpose( ) )
			self.W = (Z / Z.sum( )).flatten( 1 )
			self.Q = i
			# print self.finalclassifer(i),'==========='
			if self.finalclassifer( i ) == 0:
				
				print
				i + 1, " weak classifier is enough to  make the error to 0"
				break
	
	def finalclassifer ( self, t ):
		'''
			the 1 to t weak classifer come together
		'''
		self.sums = self.sums + self.G[t].pred( self.X ).flatten( 1 ) * self.alpha[t]
		# print self.sums
		pre_y = sign( self.sums )
		# sums=np.zeros(self.y.shape)
		# for i in range(t+1):
		#	sums=sums+self.G[i].pred(self.X).flatten(1)*self.alpha[i]
		#	print sums
		# pre_y=sign(sums)
		t = (pre_y != self.y).sum( )
		return t
	
	def pred ( self, test_set ):
		sums = np.zeros( self.y.shape )
		for i in range( self.Q + 1 ):
			sums = sums + self.G[i].pred( self.X ).flatten( 1 ) * self.alpha[i]
		# print sums
		pre_y = sign( sums )
		return pre_y

if __name__ == "__main__":
	
	features = ['Age', 'smoking index', 'FEV1/FVC%', 'FEV1%Pred', 'AST', 'NT-proBNP', 'hs-cTn',
	            'OI', 'PCO2', 'CRP', 'HGB', 'PAHD']
	datadf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02汇总.csv" )
	traindf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Trainset.csv" )[features].dropna( )
	testdf = pd.read_csv( "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/02Testset.csv" )[features].dropna( )
	# target = "PASP"
	target = "PAHD"
	# print( traindf.isnull( ).sum( ).sort_values( ascending = False ) )
	
	# ===========================数据集划分===============================
	from sklearn.utils import shuffle
	
	traindf = shuffle( traindf )
	
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
	# X_train[num_cols] = mms.fit_transform( X_train[num_cols] )
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
	# X_test[num_cols] = mms.fit_transform( X_test[num_cols] )
	
	# y_test= testdf[target].values.reshape( -1, 1 )
	# y_test = mms.fit_transform( y_test )
	y_test = testdf[target]
	
	y_test = led.fit_transform( y_test.values.ravel( ) )

	# w1为初始权重，所有样本的初始权重相等
	w1 = np.full( shape = 10, fill_value = 0.1 )