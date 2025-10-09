# Todo 相关系数
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.width', 10000)

filepath = "/Users/gyw/Desktop/COPD/COPD_GOLD_fufill.csv"
data = pd.read_csv(filepath).copy()
labels = "Ornotfrequent,Hospitaldays,GOLD".split(",")

def correlation(label):
    labels = []
    values = []
    df = data.drop([label], axis=1)
    for col in df.columns:
        labels.append(col)
        values.append(np.corrcoef(df[col], data[label], values)[0, 1])
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')
    ind = np.arange(len(labels))
    width = 1
    fig, ax = plt.subplots(figsize=(30, 30))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='blue')
    ax.set_yticks(ind)
    ax.set_yticklabels(
        corr_df.col_labels.values,
        rotation='horizontal',
        fontsize=30)
    ax.set_xlabel('Correlation coefficient', fontsize=30)
    ax.set_title(
        'Correlation coefficient of the variables--' +
        label,
        fontsize=50)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    for i in labels:
        correlation(i)