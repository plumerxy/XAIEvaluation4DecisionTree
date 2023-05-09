import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


if __name__ == '__main__':
    # ro.r("install.packages('C50')")
    C50 = importr('C50')
    C5_0 = ro.r('C5.0')

    # 文件路径
    train_file_path = "datasets/wine_train.csv"
    test_file_path = "datasets/wine_test.csv"
    blackbox_file_path = "blackbox/mlp_wine.model"
    blackbox_type = "RF"

    # 读取训练集
    x_train_df = pd.read_csv(train_file_path)
    x_train = np.array(x_train_df.iloc[:, :-1])

    # 黑盒
    blackbox = joblib.load(blackbox_file_path)
    if blackbox_type == "MLP":
        mms = MinMaxScaler()
        # mlp才需要的归一化，因为我对mlp的输入特征做了归一化处理
        mms_data_train = mms.fit_transform(x_train)
        y_train = blackbox.predict(mms_data_train)
    else:
        y_train = blackbox.predict(x_train)

    # 处理数据
    x_train_df['label'] = x_train_df['label'].apply(str)
    y = np.array(y_train.tolist())
    del x_train_df['label']

    pandas2ri.activate()
    # base = importr('base')
    # x_train_df = base.summary(x_train_df)
    x_train_df = pandas2ri.py2ri(x_train_df)
    y = ro.FactorVector(y)

    # 调用决策树模型
    model = C50.C5_0(x_train_df, y)
    print(ro.r.summary(model))

