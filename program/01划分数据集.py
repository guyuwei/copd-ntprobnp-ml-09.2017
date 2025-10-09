# Todo 划分数据集 数值归一化
import pandas as pd
from sklearn.model_selection import RepeatedKFold, train_test_split, LeaveOneOut, GridSearchCV, permutation_test_score, cross_val_score
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
#设置
pd.set_option('display.max_columns', None)
pd.options.display.width = 150

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

testsize = 0.2
BASE_DIR = Path(__file__).parent.parent
data_path = BASE_DIR / "Resources/"
output_path = BASE_DIR / "Output/"

copd_file="/Users/gyw/Desktop/Project/2025/NT-proBNP～COPD/Resources/00_DataClean_无心律失常.csv"
copd_combind_file="/Users/gyw/Desktop/Project/2025/NT-proBNP～COPD/Resources/00_DataClean_合并心律失常.csv"

Train_set=output_path/"01_Trainset.csv"
Test_set=output_path/"01_Testset.csv"
All_data=output_path/"01_所有病人汇总.csv"
if __name__ == '__main__':
	copd_df = pd.read_csv(copd_file)
	copd_combined_arrhythmia_df = pd.read_csv(copd_combind_file)
	print(copd_df.info())
	print(copd_combined_arrhythmia_df.info())
	alldata = pd.concat( [copd_df, copd_combined_arrhythmia_df] )
	X = alldata.drop( ["GOLD(1-4)"], axis = 1 )
	y = alldata["GOLD(1-4)"]
	# 划分训练集 测试集
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = testsize)
	Trainset = X_train
	Trainset["GOLD(1-4)"] = y_train
	Trainset["Group"] = "Training_Set"
	Testest = X_test
	Testest["GOLD(1-4)"]= y_test
	Testest["Group"] = "Testing_Set"
	all = pd.concat( [Trainset, Testest] )


	#
	all.to_csv(All_data, header = True, index = False )
	Trainset.to_csv( Train_set, header = True, index = False )
	Testest.to_csv( Test_set, header = True, index = False )