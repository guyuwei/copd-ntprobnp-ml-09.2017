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


class kfolder( object ):
	def __init__ ( self, dataset ):
		self.dataset = dataset
		self.length = len( dataset )
	
	def get_index ( self, N ):
		# 获取数据集长度
		num = len( self.dataset )
		# 获取分割块内数据平均数量
		div = int( np.floor( num / N ) )
		res = num % N
		sep_poi = N - res
		index_lst = []
		# 循环分割数据集
		for factor in range( 1, N + 1 ):
			if factor <= sep_poi:
				i_left = (factor - 1) * div
				i_right = factor * div
			elif factor > sep_poi:
				i_left = sep_poi * div + (factor - sep_poi - 1) * (div + 1)
				i_right = sep_poi * div + (factor - sep_poi) * (div + 1)
			else:
				raise "ERROR: parts not matched."
			index_lst.append( [i_left, i_right] )
		return index_lst
	
	def k_folder_split ( self, K_value, shuffle = False ):
		index_lst = self.get_index( K_value )
		data = self.dataset
		# 1 np.random.shuffle(x)：在原数组上进行，改变自身序列，无返回值
		# 2 np.random.permutation(x)：不在原数组上进行，返回新的数组，不改变自身数组
		# 以上两种函数只对第一维数据乱序处理。
		if shuffle:
			data = np.array( data )
			# 根据时间产生随机数，不固定seed
			np.random.seed( )
			data = np.random.permutation( data )
		else:
			data = np.array( data )
		for ind in index_lst:
			i_left, i_right = ind[0], ind[1]
			data_test = data[i_left:i_right]
			data_test_left = data[0:i_left]
			data_test_right = data[i_right:]
			data_rest = np.concatenate( (data_test_left,
			                             data_test_right), axis = 0 )
			yield data_test, data_rest


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
	def split_data_by_ratio ( self, test_size, random_state, flag ):
		train_set, test_set = train_test_split( self.dataset, test_size = test_size,
		                                        random_state = random_state, shuffle = True )
		return train_set, test_set
	
	def split_data ( self, train_set, test_set, flag ):
		col_idx = self.target_col
		train_col = self.train_col
		train_set = np.array( train_set )
		test_set = np.array( test_set )
		# flag默认为True，适用于RF,SVM,XGBoost; False适用于PLSR
		if flag:
			train_data = train_set[:, train_col[0]:train_col[1]]
			train_label = train_set[:, col_idx]
			test_data = test_set[:, train_col[0]:train_col[1]]
			test_label = test_set[:, col_idx]
		else:
			train_data = train_set[:, train_col[0]:train_col[1]]
			train_label = train_set[:, [col_idx]]
			test_data = test_set[:, train_col[0]:train_col[1]]
			test_label = test_set[:, [col_idx]]
		return np.array( train_data, dtype = 'float64' ), np.array( train_label, dtype = 'float64' ), np.array( test_data, dtype = 'float64' ), np.array( test_label, dtype = 'float64' )
	
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
			model = svm.LinearSVR( epsilon = 0.0, tol = 0.0001, C = n_com,
			                       loss = 'epsilon_insensitive', fit_intercept = True,
			                       intercept_scaling = 1.0, dual = True, verbose = 0,
			                       random_state = 0, max_iter = 1000 )
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
	
	def model_train ( self, model, train_set, n_com, flag ):
		modelname = self.modelname
		K = 10
		iter_data = kfolder( train_set )
		dataset = iter_data.k_folder_split( K_value = K, shuffle = True )
		count = 1
		for data_valid, data_train in dataset:
			print( f' {modelname:>8} training: {count} / {K}' )
			train_data, train_label, test_data, test_label = self.split_data(
					data_train, data_valid, flag )
			model.fit( train_data, train_label )
			pred_label = model.predict( test_data )
			# print(type(test_label.dtype),type(pred_label.dtype))
			
			slope, intercept, r_value, p_value, std_err = stats.linregress(
					test_label, pred_label )
			std_measure = np.std( test_label, ddof = 1 )
			rmse = self.get_rmse( test_label, pred_label )
			RPD = std_measure / rmse
			r_squared = r_value ** 2
			PRE_POINUM = train_data.shape[0]
			AIC = PRE_POINUM * np.log( rmse ) + 2 * n_com
			print( f'RMSE:{rmse}, R2:{r_squared}, RPD:{RPD}, AIC:{AIC}, pvalue:{p_value}' )
			with open( r'/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/' + '/train_log.txt', 'a' ) as trainfile:
				trainfile.write( 'training accuracy: RMSE:' + str( rmse ) + ',R2:' + str( r_squared ) + ',RPD:' + str( RPD ) + ',AIC:' + str( AIC ) + ',pvalue:' + str( p_value ) + '\n' )
			count += 1
		return model
	
	def get_para ( self, n_com, test_size, random_state ):
		my_model, flag = self.model_flag( n_com )
		train_set, test_set = self.split_data_by_ratio(
				test_size, random_state, flag )
		
		# 下面这个数据分割用于测试集验证，与训练模型无关
		train_data, _, test_data, test_label = self.split_data(
				train_set, test_set, flag )
		# my_model.fit(train_data,train_label)
		# 使用K折交叉验证训练模型，并返回
		my_model = self.model_train( my_model, train_set, n_com, flag )
		pred_label = my_model.predict( test_data )
		# flag=False:PLSR
		if not flag:
			test_label = np.squeeze( test_label )
			pred_label = np.squeeze( pred_label )
		slope, intercept, r_value, p_value, std_err = stats.linregress(
				test_label, pred_label )
		std_measure = np.std( test_label, ddof = 1 )
		rmse = self.get_rmse( test_label, pred_label )
		RPD = std_measure / rmse
		r_squared = r_value ** 2
		PRE_POINUM = train_data.shape[0]
		AIC = PRE_POINUM * np.log( rmse ) + 2 * n_com
		return rmse, RPD, AIC, r_squared, slope, p_value, test_label, pred_label


def cross_validation_test ( model_name, xlsfile, sheet, train_col, target_col, n_com ):
	data_proc = models_proc( xlsfile, sheet, train_col, target_col, model_name )
	# n_com_range_now = n_com_range[model_name]
	rmse, rpd, aic, r_squared, slope, p_value, test_label, pred_label = data_proc.get_para(
			n_com, test_size = 0.2, random_state = 1 )
	print( f'testing accuracy:\n RMSE:{rmse}, R2:{r_squared}, RPD:{rpd}, AIC:{aic}, pvalue:{p_value}' )
	with open( r'/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/' + '/prediction_result.txt', 'a' ) as txtfile:
		txtfile.write( 'testing accuracy: RMSE:' + str( rmse ) + ',R2:' + str( r_squared ) + ',RPD:' + str( rpd ) + ',AIC:' + str( aic ) + ',pvalue:' + str( p_value ) + '\n' )
		txtfile.write( 'test_label' + ',' + 'pred_label' + '\n' )
		for i in range( len( test_label ) ):
			txtfile.write( str( test_label[i] ) + ',' + str( pred_label[i] ) + '\n' )


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
	plt.savefig( 'result' + tag + '.png' )
	print( f' {tag:>8} DONE!' )


def draw_plot_all ( model_name, xlsfile, sheet, train_col, target_col, n_com_range ):
	data_proc = models_proc( xlsfile, sheet, train_col, target_col, model_name )
	n_com_range_now = n_com_range[model_name]
	rmse_lst = []
	rpd_lst = []
	aic_lst = []
	r_squared_lst = []
	slope_lst = []
	n_com_lst = []
	for n_com in range( n_com_range_now[0], n_com_range_now[1], n_com_range_now[2] ):
		print( 'Epoch {:>8} (start,end,now):{:>4},{:>4},{:>4} 正在运行，请勿关闭，谢谢！电话：15716365753'.format(
				model_name, n_com_range_now[0], n_com_range_now[1], n_com ) )
		rmse, rpd, aic, r_squared, slope = data_proc.get_para(
				n_com, test_size = 0.2, random_state = 1 )
		n_com_lst.append( n_com )
		rmse_lst.append( rmse )
		rpd_lst.append( rpd )
		aic_lst.append( aic )
		r_squared_lst.append( r_squared )
		slope_lst.append( slope )
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
	train_col = [0,35]  # 训练数据列，包左不包右，计数从0开始
	target_col = 36  # 预测的是哪一列
	# 不同模型对应的训练参数，核的数量
	n_com_range = {
			'RF' : [120, 151, 10], 'PLSR': [2, 7, 1],
			'SVM': [1, 3, 1], 'XGBOOST': [60, 121, 10]
			}
	model_lst = ['RF', 'PLSR', 'SVM', 'XGBOOST']
	
	cross_validation_test( 'RF', filename, Sheet, train_col, target_col, n_com = 200 )
	
	p1 = Process(target=draw_plot_all, args=(model_lst[0],filename,Sheet,train_col,target_col,n_com_range, ))
	p2 = Process(target=draw_plot_all, args=(model_lst[1],filename,Sheet,train_col,target_col,n_com_range, ))
	p3 = Process(target=draw_plot_all, args=(model_lst[2],filename,Sheet,train_col,target_col,n_com_range, ))
	p4 = Process(target=draw_plot_all, args=(model_lst[3],filename,Sheet,train_col,target_col,n_com_range, ))
	p1.start()
	p2.start()
	p3.start()
	p4.start()
	p1.join()
	p2.join()
	p3.join()
	p4.join()
	
	print( 'Successfully!' )
	t_end = time.time( )
	print( '-----------------------------------------------------------' )
	print( 'Strat time: ', time.asctime( time.localtime( t_start ) ) )
	print( 'End time: ', time.asctime( time.localtime( t_end ) ) )
	print( 'Total time: {} hours'.format( round( (t_end - t_start) / 3600, 4 ) ) )