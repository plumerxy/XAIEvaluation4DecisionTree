import numpy as np
import pandas as pd

df = pd.read_csv('casestudy.csv')
df.set_index(['XAI'], inplace=True)
# print(df.shape)
# print(df.head(5))

# 数据标准化
m1 = ['APL', 'Node']
# m1 = ['SUBTREE', 'ATTR']
df = df[m1]
# print(df.head(5))
for i in m1:
    Max = np.max(df[i])
    Min = np.min(df[i])
    df[i] = (Max - df[i])/(Max - Min)
print(df)

# 求指标权重
def entropy(df):
    df_new = pd.DataFrame()
    for column in df.columns:
        sigma_xij = sum(df[column])
        df_new[column] = df[column].apply(lambda x_ij: x_ij / sigma_xij if x_ij / sigma_xij != 0 else 1e-6)
    return df_new

df_new = entropy(df)

# 求熵值Hi
k = 1 / np.log(20)
h_j = (-k) * np.array([sum([pij*np.log(pij) for pij in df_new[column]]) for column in df_new.columns])
h_js = pd.Series(h_j, index=df_new.columns, name='entropy')
# 求差异系数
df_new1 = pd.Series(1-h_j, index=df_new.columns, name='difference')
# 权重
df_weight = df_new1 / sum(df_new1)
df_weight.name = 'weight'
print(df_weight)

weight = np.array(df_weight).reshape(1, -1)
df = df * weight
df['sum'] = df.apply(lambda x: x.sum(), axis=1)
print(df)

