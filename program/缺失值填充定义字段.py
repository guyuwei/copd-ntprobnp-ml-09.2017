# Todo 数据清洗
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

pd.set_option( 'display.max_rows', None )
pd.set_option( 'display.width', None )
pd.set_option( 'display.max_columns', None )
from sklearn import preprocessing
import numpy as np
from fancyimpute import IterativeImputer as MICE
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings( "ignore" )
cfile = "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/副本COPD合并心率失常1008.xlsx"
savepath = "/Users/gyw/Desktop/Project/慢阻肺合并症-gyw/"
test_size = 0.2
cols = ['Gender', 'Age', 'smoking index', 'BMI', 'COPD course', 'HTN', 'DM',
        'CHD', 'FEV1/FVC%', 'FEV1-BEST', 'FVCpred', 'FEV1%Pred', 'FVC%Pred',
        'RV/TLC%', 'DLCO%pred', 'GOLD1-4', 'ALT', 'AST', 'Cr', 'K', 'Na', 'Cl',
        'NT-proBNP', 'hs-cTn', 'D-D', 'CA-125', 'PH', 'OI', 'PCO2', 'CRP',
        'WBC', 'NEU%', 'EOS%', 'HGB', 'PLT', 'PASP', 'COPD&PN', 'PAC', 'PVC',
        'RBBB', 'SVT', 'AF', 'LBBB', 'HF', 'AdmissionTime', 'DischargeTime',
        'Daysofhospitalization', 'MedicalExpenses', 'Treatment', 'Outcomes']

# bool特征
tars = ['COPD&PN', 'PAC', 'PVC', 'RBBB', 'SVT',
        'AF', 'LBBB', 'HF']
# # c特征



def arctan_normalization ( X ):
	# 归一化
	x = np.asarray( X ).reshape( -1, 1 )
	min_max_scaler = preprocessing.MinMaxScaler( )
	x_minmax = min_max_scaler.fit_transform( x )
	return x_minmax


def df_arctan ( x, lab ):
	# TODO:对x中的labels列归一化
	y = x.copy( )
	y[lab] = arctan_normalization( x[lab] )
	return y


def one_hot ( x, col, col_class ):
	y = x.copy( )
	labels = y[col]
	Label_class = col_class
	one_hot_label = np.array( [[int( i == int( labels[j] ) ) for i in range( Label_class )] for j in range( len( labels ) )] )
	x[col] = one_hot_label
	return x


def checkparas ( data ):
	# TODO:check col.isnull
	df = data.copy( )
	uncomplete_cols = { }
	target_cols = []
	for col in df.columns:
		if len( df[df[col].isnull( )] ) != 0:
			uncomplete_cols[col] = len( df[df[col].isnull( )] )
			target_cols.append( col )
	print( "含有缺失值的列:", uncomplete_cols )
	return target_cols


def filldata ( data ):
	df_complete = data.copy( )
	for col in data:
		data_train_numeric = data[[col]]
		
		data_complete = MICE( ).fit_transform( data_train_numeric )
		# data_complete=round(data_complete,2)
		df_complete[col] = data_complete
	# print(df_complete)
	
	return df_complete


def parafufill ( data, target ):
	# TODO:fufill nan with  prediction by rf
	fillc = data[target].copy( )
	df = data.drop( columns = [target], axis = 1 )
	df_0 = SimpleImputer(
			missing_values = np.nan,
			strategy = 'mean' ).fit_transform( df )
	ytrain = fillc[fillc.notnull( )]
	ytest = fillc[fillc.isnull( )]
	xtrain = df_0[ytrain.index, :]
	xtest = df_0[ytest.index, :]
	rfc = RandomForestRegressor( n_estimators = 200 ).fit( xtrain, ytrain )
	y_predict = rfc.predict( xtest )
	data.loc[ytest.index, target] = y_predict
	
	return data


if __name__ == "__main__":
	df = pd.read_excel( cfile, 'modify' )

	
	# 数值型变量名
	num_cols = ['Age', 'smoking index', 'BMI', 'COPD course', 'FEV1/FVC%', 'FEV1-BEST', 'FVCpred', 'FEV1%Pred', 'FVC%Pred',
	                         'RV/TLC%', 'DLCO%pred', 'ALT', 'AST', 'Cr', 'K', 'Na', 'Cl',
	                         'NT-proBNP', 'hs-cTn', 'D-D', 'CA-125', 'PH', 'OI', 'PCO2', 'CRP',
	                         'WBC', 'NEU%', 'EOS%', 'HGB', 'PLT', 'PASP']
	# 分类型变量名
	cat_cols = ['HTN', 'DM', 'CHD', 'Gender', 'COPD&PN', 'PAC', 'PVC', 'RBBB', 'SVT',
	            'AF', 'LBBB', 'HF', 'GOLD1-4']
	drop=[]
	sum=df.isnull( ).sum( )
	for i,j in enumerate(sum):
		print(df.columns[i],j)
		if j >150:
			drop.append(df.columns[i])
	print(drop)
	
	# ======================特征编码============================


	# data = pd.get_dummies(df, columns = cat_cols )
	print( df.head( ) )
	data= filldata( df.drop(columns = drop,axis = 1) )
	data["GOLD1-4"]=df["GOLD1-4"]
	

	print( data.isnull( ).sum( ).sort_values( ascending = False ) )
	print(data.columns)
	data.to_csv( savepath + "01COPD__Complete.csv", header = True, index = False )