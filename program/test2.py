# -*- coding: utf-8 -*-
"""
任何问题联系邮箱：chinesevoice@163.com
@author: cz
"""

from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from multiprocessing import Process
from multiprocessing import Manager
import time
import xgboost as xgb
import pandas as pd
import copy
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os, gc

xgb.set_config( verbosity = 0 )


class models_proc( object ):
	def __init__ ( self, xlsfile, sheet, train_col, target_col, modelname ):
		self.filename = xlsfile
		self.sheet = sheet
		self.train_col = train_col
		self.target_col = target_col
		# self.n_com = n_com
		self.modelname = modelname
		data_init = pd.read_excel( self.filename, sheet_name = self.sheet )
		self.dataset = copy.deepcopy( data_init )
		data_init = None
		del data_init
		gc.collect( )
	
	# 定义读取数据的函数
	def split_data ( self, test_size, random_state, flag ):
		col_idx = self.target_col
		train_col = self.train_col
		train_set, test_set = train_test_split( self.dataset, test_size = test_size,
		                                        random_state = random_state, shuffle = True )
		# flag默认为True，适用于RF,SVM,XGBoost; False适用于PLSR
		if flag:
			train_data = np.array( train_set.iloc[:, train_col[0]:train_col[1]] )
			train_label = np.array( train_set.iloc[:, col_idx] )
			test_data = np.array( test_set.iloc[:, train_col[0]:train_col[1]] )
			test_label = np.array( test_set.iloc[:, col_idx] )
		else:
			train_data = np.array( train_set.iloc[:, train_col[0]:train_col[1]] )
			train_label = np.array( train_set.iloc[:, [col_idx]] )
			test_data = np.array( test_set.iloc[:, train_col[0]:train_col[1]] )
			test_label = np.array( test_set.iloc[:, [col_idx]] )
		return train_data, train_label, test_data, test_label
	
	def model_flag ( self, n_com ):
		model_name = self.modelname
		if model_name == 'PLSR':
			model = PLSRegression( n_components = n_com, scale = True )
			flag = False
		elif model_name == 'RF':
			model = RandomForestRegressor( n_estimators = n_com, max_depth = None,
			                               random_state = 0, bootstrap = True,
			                               oob_score = True )
			flag = True
		elif model_name == 'SVM':
			# 注意：这里得n_com参数实际上没有用到SVM模型参数里，所以出来的图结果都一样
			# model = svm.LinearSVR(C=1.0,kernel='linear', degree=3, gamma='auto',
			#             coef0=0.0,tol=0.001)
			# model = svm.LinearSVR(epsilon=0.0, tol=0.0001, C=n_com,
			#                       loss='epsilon_insensitive', fit_intercept=True,
			#                       intercept_scaling=1.0, dual=True, verbose=0,
			#                       random_state=0, max_iter=1000)
			model = svm.SVR( C = 1.0, kernel = 'rbf', degree = n_com, gamma = 'auto',
			                 coef0 = 0.0, shrinking = True, tol = 0.0001, cache_size = 200,
			                 verbose = False, max_iter = -1 )
			flag = True
		elif model_name == 'XGBOOST':
			model = xgb.XGBRegressor( objective = 'reg:linear', colsample_bytree = 0.3,
			                          learning_rate = 0.1, max_depth = 5, alpha = 10,
			                          n_estimators = n_com )
			flag = True
		else:
			raise 'WRONG MODEL PARA!'
		return model, flag
	
	def get_rmse ( self, test_target, pred_data ):
		error = test_target - pred_data
		error_2 = error ** 2
		error_sum = error_2.sum( )
		rmse = (error_sum / len( test_target )) ** 0.5
		return rmse
	
	def get_para ( self, n_com, test_size, random_state ):
		my_model, flag = self.model_flag( n_com )
		train_data, train_label, test_data, test_label = self.split_data(
				test_size, random_state, flag )
		# print('=======RUN HERE!!!=======')
		my_model.fit( train_data, train_label )
		pred_label = my_model.predict( test_data )
		# print('=======RUN HERE!!!=======')
		# flag=False:PLSR
		if not flag:
			test_label = np.squeeze( test_label )
			pred_label = np.squeeze( pred_label )
		# print('test_label',test_label.shape)
		slope, intercept, r_value, p_value, std_err = stats.linregress(
				test_label, pred_label )
		std_measure = np.std( test_label, ddof = 1 )
		rmse = self.get_rmse( test_label, pred_label )
		RPD = std_measure / rmse
		r_squared = r_value ** 2
		PRE_POINUM = train_data.shape[0]
		AIC = PRE_POINUM * np.log( rmse ) + 2 * n_com
		return rmse, RPD, AIC, r_squared, slope


def draw_plot ( rmse_lst, rpd_lst, aic_lst, r_squared_lst, slope_lst, n_com_lst, tag = 'PLSR' ):
	fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots( 5, 1, figsize = (5, 10) )
	plt.tight_layout( pad = 1.2 )
	# ax1.set_title('FILE:'+filename[:-4]+' Samples:{}'.format(tag))
	ax1.set_title( tag )
	ax1.plot( n_com_lst, rmse_lst )
	ax1.set_ylabel( 'RMSE' )
	ax1.set_xticks( n_com_lst )
	ax1.axhline( min( rmse_lst ), color = 'red', linestyle = '--' )
	ax1.text( min( n_com_lst ), (min( rmse_lst ) + max( rmse_lst )) / 2,
	          r'$MIN_{RMSE}=$' + str( round( min( rmse_lst ), 4 ) ),
	          color = 'red', fontsize = 14 )
	ax1.grid( )
	ax2.plot( n_com_lst, rpd_lst )
	ax2.set_ylabel( 'RPD' )
	ax2.set_xticks( n_com_lst )
	ax2.axhline( max( rpd_lst ), color = 'red', linestyle = '--' )
	ax2.text( min( n_com_lst ), (min( rpd_lst ) + max( rpd_lst )) / 2,
	          r'$MAX_{RPD}=$' + str( round( max( rpd_lst ), 4 ) ),
	          color = 'red', fontsize = 14 )
	ax2.grid( )
	ax3.plot( n_com_lst, aic_lst )
	ax3.set_ylabel( 'AIC' )
	ax3.set_xticks( n_com_lst )
	ax3.axhline( min( aic_lst ), color = 'red', linestyle = '--' )
	ax3.text( min( n_com_lst ), (min( aic_lst ) + max( aic_lst )) / 2,
	          r'$MIN_{AIC}=$' + str( round( min( aic_lst ), 4 ) ),
	          color = 'red', fontsize = 14 )
	ax3.grid( )
	ax4.plot( n_com_lst, r_squared_lst )
	ax4.set_ylabel( 'R²' )
	ax4.set_xticks( n_com_lst )
	ax4.axhline( max( r_squared_lst ), color = 'red', linestyle = '--' )
	ax4.text( min( n_com_lst ), (min( r_squared_lst ) + max( r_squared_lst )) / 2,
	          r'$MAX_{R^2}=$' + str( round( max( r_squared_lst ), 4 ) ),
	          color = 'red', fontsize = 14 )
	ax4.grid( )
	ax5.plot( n_com_lst, slope_lst )
	ax5.set_ylabel( 'SLOPE' )
	ax5.set_xticks( n_com_lst )
	ax5.set_xlabel( tag )
	ax5.axhline( max( slope_lst ), color = 'red', linestyle = '--' )
	ax5.text( min( n_com_lst ), (min( slope_lst ) + max( slope_lst )) / 2,
	          r'$MAX_{SLOPE}=$' + str( round( max( slope_lst ), 4 ) ),
	          color = 'red', fontsize = 14 )
	ax5.grid( )
	# plt.show()
	plt.savefig( r'F:\NIGLAS_DrZYC\cz\DL4AVB_ADDPARA' + '/result' + tag + '.png' )
	# plt.savefig('result' + tag + '.png')
	print( f' {tag:>8} DONE!' )


def draw_plot_all ( model_name, xlsfile, sheet, train_col, target_col, n_com_range ):
	data_proc = models_proc( xlsfile, sheet, train_col, target_col, model_name )
	# print('=======RUN HERE!!!=======')
	n_com_range_now = n_com_range[model_name]
	rmse_lst = []
	rpd_lst = []
	aic_lst = []
	r_squared_lst = []
	slope_lst = []
	n_com_lst = []
	for n_com in range( n_com_range_now[0], n_com_range_now[1], n_com_range_now[2] ):
		print( 'Epoch {:>8} (start,end,now):{:>4},{:>4},{:>4} 正在运行，请勿关闭，谢谢！'.format(
				model_name, n_com_range_now[0], n_com_range_now[1], n_com ) )
		rmse, rpd, aic, r_squared, slope = data_proc.get_para(
				n_com, test_size = 0.2, random_state = 1 )
		# print('=======RUN HERE!!!=======')
		n_com_lst.append( n_com )
		rmse_lst.append( rmse )
		rpd_lst.append( rpd )
		aic_lst.append( aic )
		r_squared_lst.append( r_squared )
		slope_lst.append( slope )
	# print('=======RUN HERE!!!=======')
	draw_plot( rmse_lst, rpd_lst, aic_lst, r_squared_lst, slope_lst, n_com_lst, tag = model_name )


# 多进程调用函数制图
if __name__ == '__main__':
	# =============设置参数================
	manager = Manager( )
	# lock = manager.Lock()
	t_start = time.time( )
	# cwd_path = os.getcwd()
	path = r'/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/'
	xlsname = '副本COPD合并心率失常1008.xlsx'
	Sheet = 'complete'
	filename = os.path.join( path, xlsname )
	train_col = [0, 35]  # 训练数据列，包左不包右，计数从0开始
	target_col = 36  # 预测的是哪一列
	# 不同模型对应的训练参数，核的数量
	n_com_range = {
			'RF' : [160, 201, 10], 'PLSR': [10, 101, 10],
			'SVM': [3, 10, 1], 'XGBOOST': [10000, 15001, 500]
			}
	model_lst = ['RF', 'PLSR', 'SVM', 'XGBOOST']
	
	p_test = Process( target = draw_plot_all, args = ('XGBOOST', filename, Sheet, train_col, target_col, n_com_range,) )
	p_test.start( )
	p_test.join( )
	
	# p1 = Process(target=draw_plot_all, args=(model_lst[0],filename,Sheet,train_col,target_col,n_com_range, ))
	# p2 = Process(target=draw_plot_all, args=(model_lst[1],filename,Sheet,train_col,target_col,n_com_range, ))
	# p3 = Process(target=draw_plot_all, args=(model_lst[2],filename,Sheet,train_col,target_col,n_com_range, ))
	# p4 = Process(target=draw_plot_all, args=(model_lst[3],filename,Sheet,train_col,target_col,n_com_range, ))
	# p1.start()
	# p2.start()
	# p3.start()
	# p4.start()
	# p1.join()
	# p2.join()
	# p3.join()
	# p4.join()
	
	print( 'Successfully!' )
	t_end = time.time( )
	print( '-----------------------------------------------------------' )
	print( 'Strat time: ', time.asctime( time.localtime( t_start ) ) )
	print( 'End time: ', time.asctime( time.localtime( t_end ) ) )
	print( 'Total time: {} hours'.format( round( (t_end - t_start) / 3600, 4 ) ) )