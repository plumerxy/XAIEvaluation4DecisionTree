from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

"""读取训练集和测试集"""
df = pd.read_csv('datasets/wine_train.csv')
x_train = np.array(df.iloc[:, :-1])
y_train = np.array(df.iloc[:, -1]).reshape(-1,)
df = pd.read_csv('datasets/wine_test.csv')
x_test = np.array(df.iloc[:, :-1])
y_test = np.array(df.iloc[:, -1]).reshape(-1,)

# mms = MinMaxScaler()
# mms_data_train = mms.fit_transform(x_train)
# mms_data_test = mms.fit_transform(x_test)

"""构建MLP"""
clf = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(64, 32, 16), max_iter=1000)
clf.fit(x_train, y_train)
print('层数：%s，输出单元数量：%s' % (clf.n_layers_, clf.n_outputs_))
print('MLP模型得分:{:.2f}'.format(clf.score(x_test, y_test)))
joblib.dump(clf, "blackbox/mlp_wine.model")


"""构建随机森林"""
rfc = RandomForestClassifier(n_estimators=100,random_state=90)
rfc.fit(x_train, y_train)
print('RF模型得分:{:.2f}'.format(rfc.score(x_test, y_test)))
joblib.dump(rfc, "blackbox/rf_wine.model")
