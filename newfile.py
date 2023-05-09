import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


csv_data = pd.read_csv("wine.csv", low_memory=False)  # 防止弹出警告
feature_name = csv_data.columns[:-1]
X = np.array(csv_data)[:, :-1]
y = np.array(csv_data)[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
train = pd.DataFrame(X_train, columns=feature_name)
train['label'] = y_train.astype(int)
test = pd.DataFrame(X_test, columns=feature_name)
test['label'] = y_test.astype(int)


train.to_csv('wine_train.csv', index=False)
test.to_csv("wine_test.csv", index=False)