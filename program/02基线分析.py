import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings('ignore')
pd.set_option( 'display.width', 220 )
pd.set_option('display.max_columns',10000)
pd.set_option('display.max_colwidth',80)

Traindf=pd.read_csv('/Users/gyw/Desktop/Project/2024/NT-proBNP～COPD/Output/01_Trainset.csv')
Testdf=pd.read_csv("/Users/gyw/Desktop/Project/2024/NT-proBNP～COPD/Output/01_Testset.csv")
alldf=pd.read_csv("/Users/gyw/Desktop/Project/2024/NT-proBNP～COPD/Output/01_所有病人汇总.csv")
X_train=Traindf.drop(['GOLD(1-4)'],axis=1)
y_train=Traindf["GOLD(1-4)"]
X_test=Testdf.drop(["GOLD(1-4)"],axis=1)
y_test=Testdf["GOLD(1-4)"]
# 定义变量类型
bool_index = "CAD,DM,GOLD(1-4),GENDER,HF,HTN,OUTCOMES,SMOKINGINDEX,TREATMENT".split(",")
int_index = "ALT,AST,AGE,OI,PASP,PCO2,RV/TLC%,DAYSOFHOSPITALIZATION".split(",")
float_index = "BMI,CL,CR,D-D,FEV1-BEST,FVCPRED,K,MEDICALEXPENSES,NT-PROBNP,NA,PH,HS-CTN,DLCO%PRED,FEV1%PRED,FEV1/FVC%,FVC%PRED".split(
    ',')
AF_index = "AF,LBBB,PAC,PVC,RBBB,SVT".split(",")
#测试、训练集个数
Train_num=alldf['Group'].value_counts()["Training_Set"]
Test_num=alldf['Group'].value_counts()["Testing_Set"]
# 创建一个空的基线描述表格

baseline_table = pd.DataFrame(columns=['Variable', 'Training Set (n=%d)'%Train_num, 'Testing Set (n=%d)'%Test_num, 'Z', 'P'])

# 处理分类变量
for var in bool_index + AF_index:
    # 计算每个变量的计数和百分比
    train_counts = Traindf[var].value_counts(normalize=True) * 100
    test_counts = Testdf[var].value_counts(normalize=True) * 100

    # 卡方检验
    contingency_table = pd.crosstab(Traindf[var], Testdf[var])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    # 创建新行的 DataFrame
    new_row = pd.DataFrame({
        'Variable': [var],
        'Training Set (n=491)': [
            f"{train_counts[1] if 1 in train_counts else 0:.3f}% ({train_counts[0] if 0 in train_counts else 0:.0f})"],
        'Testing Set (n=211)': [
            f"{test_counts[1] if 1 in test_counts else 0:.3f}% ({test_counts[0] if 0 in test_counts else 0:.0f})"],
        'Z': [None],  # 分类变量不适用 Z 值
        'P': [f"{p_value:.3f}"]
    })

    # 使用 pd.concat 将新行添加到表格
    baseline_table = pd.concat([baseline_table, new_row], ignore_index=True)

# 处理连续变量
for var in int_index + float_index:
    # 计算中位数和IQR
    train_median = Traindf[var].median()
    train_iqr = Traindf[var].quantile(0.75) - Traindf[var].quantile(0.25)
    test_median = Testdf[var].median()
    test_iqr = Testdf[var].quantile(0.75) - Testdf[var].quantile(0.25)

    # Mann-Whitney U 检验
    stat, p_value = mannwhitneyu(Traindf[var], Testdf[var])

    # 创建新行的 DataFrame
    new_row = pd.DataFrame({
        'Variable': [var],
        'Training Set (n=491)': [f"{train_median:.2f} ({train_iqr:.2f})"],
        'Testing Set (n=211)': [f"{test_median:.2f} ({test_iqr:.2f})"],
        'Z': [f"{stat:.2f}"],
        'P': [f"{p_value:.3f}"]
    })

    # 使用 pd.concat 将新行添加到表格
    baseline_table = pd.concat([baseline_table, new_row], ignore_index=True)
baseline_table.style.set_properties(**{'text-align': 'left', 'white-space': 'normal'})
# 输出基线描述表格
print(baseline_table,)