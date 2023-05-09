import joblib
from chefboost import Chefboost as chef
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # 文件路径
    train_file_path = "datasets/wine_train.csv"
    test_file_path = "datasets/wine_test.csv"
    blackbox_file_path = "blackbox/mlp_wine.model"
    blackbox_type = "RF"
    # 读取训练集
    x_train_df = pd.read_csv(train_file_path)
    x_train_df = x_train_df.rename(columns={"label": "Decision"})
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
    cs = pd.Series(y_train)
    x_train_df['Decision'] = cs
    x_train_df['Decision'] = x_train_df['Decision'].apply(str)
    print(x_train_df)

    # 训练决策树
    config = {"algorithm": 'CHAID', 'enableParallelism': False}
    model = chef.fit(x_train_df, config=config)
