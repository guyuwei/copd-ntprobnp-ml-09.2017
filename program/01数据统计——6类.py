#Todo: 从预后角度描述多种心律失常疾病对COPD患者的影响
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore
import warnings
from matplotlib.font_manager import FontProperties
import seaborn as sns

# 忽略警告
warnings.filterwarnings('ignore')

# 设置 seaborn 风格
sns.set(style="whitegrid", font_scale=1.2)  # 使用 seaborn 风格
plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴线宽
plt.rcParams['figure.dpi'] = 300  # 图像分辨率
font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"  # macOS 系统
font_prop = FontProperties(fname=font_path, size=16)
plt.rcParams['font.family'] = font_prop.get_name()  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置 Pandas 显示选项
pd.set_option('display.max_columns', None)
pd.options.display.width = 200



# 统计每种心律失常的发生次数
def count_arrhythmia_patients(df):
    """
    统计并可视化 COPD 患者中每种心律失常的发生频率。
    :param df: 包含心律失常类型列的数据框
    """
    # 统计每种心律失常的发生次数
    arrhythmia_counts = {arrhythmia: df[arrhythmia].sum() for arrhythmia in AF_index}
    for arrhythmia, count in arrhythmia_counts.items():
        print(f"{arrhythmia} 发生次数: {count}")

    # 汇总结果
    result_df = pd.DataFrame({
        "心律失常类型": AF_index,
        "发生次数": [arrhythmia_counts[arrhythmia] for arrhythmia in AF_index]
    }).sort_values(by="发生次数", ascending=False)

    print("心律失常发生频率统计:\n", result_df)

    # 绘制心律失常发生频率柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        result_df["心律失常类型"],
        result_df["发生次数"],
        color=sns.color_palette("Set3"),  # 使用 seaborn 的调色板
        edgecolor="black",  # 柱状图边框颜色
        alpha=0.8  # 透明度
    )

    # 在每个柱子上添加具体值
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x 坐标：柱子中心
            height * 0.95,  # y 坐标：柱子顶部稍微向下偏移
            f"{int(height)}",  # 显示的文本
            ha="center",  # 水平对齐方式
            va="top",  # 垂直对齐方式
            fontsize=14,  # 字体大小
            color="black",  # 字体颜色
            fontproperties=font_prop  # 设置中文字体
        )

    # 设置标题和标签
    plt.title("COPD 患者心律失常发生频率", fontsize=18, pad=20, fontproperties=font_prop)
    plt.xlabel("心律失常类型", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.ylabel("发生次数", fontsize=14, labelpad=10, fontproperties=font_prop)

    # 设置刻度样式
    plt.xticks(fontsize=12, fontproperties=font_prop, rotation=0)
    plt.yticks(fontsize=12, fontproperties=font_prop)

    # 显示柱状图
    plt.tight_layout()
    plt.show()

    # 检查数据是否为空
    if not result_df.empty:
        # 绘制饼状图
        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            result_df["发生次数"],  # 数据
            labels=result_df["心律失常类型"],  # 标签
            autopct=lambda p: f"{p:.1f}%",  # 显示百分比
            startangle=90,  # 起始角度
            colors=sns.color_palette("Set3"),  # 使用 seaborn 的调色板
            textprops={"fontsize": 14, "fontproperties": font_prop},  # 字体设置
            wedgeprops={"edgecolor": "black", "linewidth": 1.5}  # 饼图边框
        )

        # 设置中文字体
        for text in texts:
            text.set_fontproperties(font_prop)
        for autotext in autotexts:
            autotext.set_fontproperties(font_prop)

        # 设置标题
        plt.title("COPD 患者心律失常发生频率分布", fontsize=16, pad=20, fontproperties=font_prop)
        plt.tight_layout()
        plt.show()


def count_severity(df):
    """
    统计并可视化 COPD 患者中每种心律失常的严重程度（OUTCOMES）分布。
    :param df: 包含心律失常类型和 OUTCOMES 列的数据框
    """
    unique_outcomes = df["OUTCOMES"].unique()
    print("OUTCOMES 列的唯一值：", unique_outcomes)

    # 统计每个心律失常中 OUTCOMES 的占比
    OUTCOMES_results = []
    for arrhythmia in AF_index:
        patients_with_arrhythmia = df[df[arrhythmia] == 1]
        outcome_counts = patients_with_arrhythmia["OUTCOMES"].value_counts().reindex(unique_outcomes, fill_value=0)
        outcome_percentages = outcome_counts / outcome_counts.sum() * 100
        result = {"心律失常类型": arrhythmia}
        for outcome in unique_outcomes:
            result[f"Outcome_{outcome}_Count"] = outcome_counts.get(outcome, 0)
            result[f"Outcome_{outcome}_Percentage"] = outcome_percentages.get(outcome, 0)
        OUTCOMES_results.append(result)

    # 转换为 DataFrame
    OUTCOMES_df = pd.DataFrame(OUTCOMES_results)
    print("心律失常严重程度统计:\n", OUTCOMES_df)

    # 绘制严重程度堆叠柱状图
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(OUTCOMES_df))
    colors = sns.color_palette("Set3")  # 使用 seaborn 的调色板
    labels = ["存活", "死亡"]  # 设置标签为存活和死亡

    for i, outcome in enumerate(unique_outcomes):
        bars = plt.bar(
            OUTCOMES_df["心律失常类型"],
            OUTCOMES_df[f"Outcome_{outcome}_Count"],
            label=labels[i],  # 使用自定义标签
            color=colors[i],
            bottom=bottom,
            edgecolor="black",  # 柱状图边框颜色
            alpha=0.8  # 透明度
        )
        bottom += OUTCOMES_df[f"Outcome_{outcome}_Count"]

        # 在柱状图上添加数值
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只显示大于 0 的数值
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{int(height)}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    fontproperties=font_prop
                )

    # 设置标题和标签
    plt.title("COPD 患者心律失常严重程度分布", fontsize=18, pad=20, fontproperties=font_prop)
    plt.xlabel("心律失常类型", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.ylabel("患者数量", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.xticks(fontsize=12, fontproperties=font_prop)
    plt.yticks(fontsize=12, fontproperties=font_prop)
    plt.legend(prop=font_prop, fontsize=12)
    plt.tight_layout()
    plt.show()


def count_treatment(df):
    """
    统计并可视化 COPD 患者中每种心律失常的治疗方式（TREATMENT）分布。
    :param df: 包含心律失常类型和 TREATMENT 列的数据框
    """
    unique_treatments = df["TREATMENT"].unique()
    print("TREATMENT 列的唯一值：", unique_treatments)

    # 统计每种心律失常中 TREATMENT 的占比
    treatment_results = []
    for arrhythmia in AF_index:
        patients_with_arrhythmia = df[df[arrhythmia] == 1]
        treatment_counts = patients_with_arrhythmia["TREATMENT"].value_counts().reindex(unique_treatments, fill_value=0)
        treatment_percentages = treatment_counts / treatment_counts.sum() * 100
        result = {"心律失常类型": arrhythmia}
        for treatment in unique_treatments:
            result[f"Treatment_{treatment}_Count"] = treatment_counts.get(treatment, 0)
            result[f"Treatment_{treatment}_Percentage"] = treatment_percentages.get(treatment, 0)
        treatment_results.append(result)

    # 转换为 DataFrame
    treatment_df = pd.DataFrame(treatment_results)
    print("心律失常治疗方式统计:\n", treatment_df)

    # 绘制治疗方式堆叠柱状图
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(treatment_df))
    colors = sns.color_palette("Set3")  # 使用 seaborn 的调色板
    labels = ["基础治疗", "气管插管/呼吸机"]  # 设置标签为基础治疗和气管插管/呼吸机

    for i, treatment in enumerate(unique_treatments):
        bars = plt.bar(
            treatment_df["心律失常类型"],
            treatment_df[f"Treatment_{treatment}_Count"],
            label=labels[i],  # 使用自定义标签
            color=colors[i],
            bottom=bottom,
            edgecolor="black",  # 柱状图边框颜色
            alpha=0.8  # 透明度
        )
        bottom += treatment_df[f"Treatment_{treatment}_Count"]

        # 在柱状图上添加数值
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只显示大于 0 的数值
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{int(height)}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    fontproperties=font_prop
                )

    # 设置标题和标签
    plt.title("COPD 患者心律失常治疗方式分布", fontsize=18, pad=20, fontproperties=font_prop)
    plt.xlabel("心律失常类型", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.ylabel("患者数量", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.xticks(fontsize=12, fontproperties=font_prop)
    plt.yticks(fontsize=12, fontproperties=font_prop)
    plt.legend(prop=font_prop, fontsize=12)
    plt.tight_layout()
    plt.show()

# 统计所有人的开销
def count_medical_expense(df):
    """
    统计和分析 COPD 患者的医疗费用（MEDICALEXPENSES）分布。
    :param df: 包含 MEDICALEXPENSES 列的数据框
    """
    if "MEDICALEXPENSES" not in df.columns:
        raise ValueError("数据集中未找到 MEDICALEXPENSES 列")

    # 描述性统计
    def descriptive_statistics():
        """计算并显示 MEDICALEXPENSES 的描述性统计量。"""
        desc_stats = df["MEDICALEXPENSES"].describe()
        print("描述性统计:\n", desc_stats)
        print("偏度:", df["MEDICALEXPENSES"].skew())
        print("峰度:", df["MEDICALEXPENSES"].kurt())

        # 箱线图
        plt.figure(figsize=(8, 6))
        plt.boxplot(df["MEDICALEXPENSES"].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor="#1f77b4"))
        plt.title("医疗费用分布箱线图", fontsize=18, pad=20, fontproperties=font_prop)
        plt.xlabel("医疗费用", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.ylabel("分布", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.xticks(fontsize=12, fontproperties=font_prop)
        plt.yticks(fontsize=12, fontproperties=font_prop)
        plt.show()

    # 分布分析
    def distribution_analysis():
        """绘制 MEDICALEXPENSES 的分布直方图和核密度估计图。"""
        # 直方图
        plt.figure(figsize=(10, 6))
        plt.hist(df["MEDICALEXPENSES"].dropna(), bins=50, color="blue", edgecolor="black", alpha=0.7)
        plt.title("医疗费用分布直方图", fontsize=18, pad=20, fontproperties=font_prop)
        plt.xlabel("医疗费用", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.ylabel("频数", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.xticks(fontsize=12, fontproperties=font_prop)
        plt.yticks(fontsize=12, fontproperties=font_prop)
        plt.show()

        # 核密度估计图
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df["MEDICALEXPENSES"].dropna(), color="blue", shade=True)
        plt.title("医疗费用核密度估计图", fontsize=18, pad=20, fontproperties=font_prop)
        plt.xlabel("医疗费用", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.ylabel("密度", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.xticks(fontsize=12, fontproperties=font_prop)
        plt.yticks(fontsize=12, fontproperties=font_prop)
        plt.show()

    # 异常值检测
    def outlier_detection():
        """检测并显示 MEDICALEXPENSES 的异常值。"""
        df["Z_score"] = zscore(df["MEDICALEXPENSES"])
        outliers = df[df["Z_score"].abs() > 3]
        print("异常值数量:", outliers.shape)

        # 异常值箱线图
        plt.figure(figsize=(8, 6))
        plt.boxplot(df["MEDICALEXPENSES"].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor="#ff7f0e"))
        plt.title("医疗费用异常值检测", fontsize=18, pad=20, fontproperties=font_prop)
        plt.xlabel("医疗费用", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.ylabel("分布", fontsize=14, labelpad=10, fontproperties=font_prop)
        plt.xticks(fontsize=12, fontproperties=font_prop)
        plt.yticks(fontsize=12, fontproperties=font_prop)
        plt.show()


    # 执行所有分析
    descriptive_statistics()
    distribution_analysis()
    outlier_detection()
#统计按疾病分类的开销
def group_analysis(df, group_by_columns):
    """
    按指定列分组，分析每组（值为 1）的医疗费用分布，并绘制所有疾病的箱线图。
    :param df: 包含数据的数据框
    :param group_by_columns: 分组列名列表
    """
    # 整体统计数据
    overall_stats = df["MEDICALEXPENSES"].describe().to_frame().T
    overall_stats.index = ["整体"]
    print("整体统计数据:\n", overall_stats)

    # 分组统计数据
    group_stats = []
    for column in group_by_columns:
        if column not in df.columns:
            print(f"数据集中未找到 {column} 列，跳过该列的分组分析")
            continue

        # 筛选当前疾病为 1 的患者
        patients_with_disease = df[df[column] == 1]

        # 检查是否有数据
        if patients_with_disease.empty:
            print(f"{column} 为 1 的患者数量为 0，跳过该列的分析")
            continue

        # 计算统计量
        stats = patients_with_disease["MEDICALEXPENSES"].describe().to_frame().T
        stats.index = [column]
        group_stats.append(stats)

    # 合并分组统计结果
    if group_stats:
        group_stats_df = pd.concat(group_stats)
        print("分组统计数据:\n", group_stats_df)
    else:
        print("没有可用的分组数据进行统计")

    # 筛选每种疾病为 1 的患者，并添加疾病类型列
    data_to_plot = []
    for column in group_by_columns:
        if column not in df.columns:
            continue

        # 筛选当前疾病为 1 的患者
        patients_with_disease = df[df[column] == 1]

        # 检查是否有数据
        if patients_with_disease.empty:
            continue

        # 添加疾病类型列
        patients_with_disease = patients_with_disease.assign(疾病类型=column)
        data_to_plot.append(patients_with_disease[["疾病类型", "MEDICALEXPENSES"]])

    # 如果没有数据，直接返回
    if not data_to_plot:
        print("没有可用的数据进行绘图")
        return

    # 合并数据
    combined_data = pd.concat(data_to_plot)

    # 绘制所有疾病的箱线图
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="疾病类型",
        y="MEDICALEXPENSES",
        data=combined_data,
        palette="Set3",
        width=0.6,
        linewidth=1.5
    )
    plt.title("不同类型心律失常患者的医疗费用分布", fontsize=18, pad=20, fontproperties=font_prop)
    plt.xlabel("疾病类型", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.ylabel("医疗费用", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.xticks(fontsize=12, fontproperties=font_prop)
    plt.yticks(fontsize=12, fontproperties=font_prop)
    plt.tight_layout()
    plt.show()
#统计发生心衰的概率
def hf_probability_by_arrhythmia(df):
    """
    计算各心律失常患者发生心衰（HF）的概率，并绘制柱状图。
    :param df: 包含心律失常类型和 HF 列的数据框
    """
    # 统计每种心律失常患者中 HF 为 1 的比例
    hf_probabilities = {}
    for arrhythmia in AF_index:
        if arrhythmia not in df.columns:
            print(f"数据集中未找到 {arrhythmia} 列，跳过该列的分析")
            continue

        # 筛选当前心律失常为 1 的患者
        patients_with_arrhythmia = df[df[arrhythmia] == 1]

        # 计算 HF 为 1 的比例
        if len(patients_with_arrhythmia) > 0:
            hf_probability = patients_with_arrhythmia["HF"].mean() * 100  # 转换为百分比
            hf_probabilities[arrhythmia] = hf_probability
            print(f"{arrhythmia} 患者发生心衰的概率: {hf_probability:.2f}%")
        else:
            print(f"{arrhythmia} 为 1 的患者数量为 0，跳过该列的分析")

    # 转换为 DataFrame
    result_df = pd.DataFrame({
        "心律失常类型": list(hf_probabilities.keys()),
        "心衰概率 (%)": list(hf_probabilities.values())
    }).sort_values(by="心衰概率 (%)", ascending=False)

    print("各心律失常患者发生心衰的概率统计:\n", result_df)

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        result_df["心律失常类型"],
        result_df["心衰概率 (%)"],
        color=sns.color_palette("Set3"),  # 使用 seaborn 的调色板
        edgecolor="black",  # 柱状图边框颜色
        alpha=0.8  # 透明度
    )

    # 在每个柱子上添加具体值
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x 坐标：柱子中心
            height * 0.95,  # y 坐标：柱子顶部稍微向下偏移
            f"{height:.2f}%",  # 显示的文本
            ha="center",  # 水平对齐方式
            va="top",  # 垂直对齐方式
            fontsize=14,  # 字体大小
            color="black",  # 字体颜色
            fontproperties=font_prop  # 设置中文字体
        )

    # 设置标题和标签
    plt.title("各心律失常患者发生心衰的概率", fontsize=18, pad=20, fontproperties=font_prop)
    plt.xlabel("心律失常类型", fontsize=14, labelpad=10, fontproperties=font_prop)
    plt.ylabel("心衰概率 (%)", fontsize=14, labelpad=10, fontproperties=font_prop)

    # 设置刻度样式
    plt.xticks(fontsize=12, fontproperties=font_prop, rotation=0)
    plt.yticks(fontsize=12, fontproperties=font_prop)

    # 显示柱状图
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    data_path = BASE_DIR / "Output/"
    all_patients_file = data_path / "01_所有病人汇总.csv"
    copd_patients = pd.read_csv(all_patients_file)


    AF_index = "AF,LBBB,PAC,PVC,RBBB,SVT".split(",")# 心律失常类型
    # Assessment of prognostic indicators
    # count_arrhythmia_patients(copd_patients)              #心律失常病人个数
    # count_severity(copd_patients)                    #严重程度统计
    # count_treatment(copd_patients)             #治疗方案统计
    # count_medical_expense(copd_patients)             #整体描述该连续数据
    # group_analysis(copd_patients, AF_index)                                   #统计按疾病分类的开销
    hf_probability_by_arrhythmia(copd_patients)     #统计发生心衰的概率


