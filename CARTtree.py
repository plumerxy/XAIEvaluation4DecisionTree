from sklearn.tree._tree import TREE_LEAF
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
import pydotplus


def getTrainOrTestData(filepath):
    # 不要第一行 标题行
    matrix = np.loadtxt(open(filepath, 'rb'), delimiter=",", skiprows=1)
    row, col = matrix.shape
    # 训练集数据（不带标签）
    data = matrix[:, :col - 1]
    # 训练集标签
    label = matrix[:, col - 1:col]
    return data, label


# 获取特征名
def getFeatureNames(filepath):
    with open(filepath, 'r') as f:
        title = f.readline()
    f.close()
    title = title.strip().split(",")
    title_num = len(title)
    title = np.array(title)
    feature_name = title[:title_num - 1]
    return feature_name


# 可视化决策树
def visualize_decisionTree(tree, path, target_names, features):
    dot_data = export_graphviz(tree, out_file=None, class_names=target_names,
                               filled=True, rounded=True, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # # graph = make_graph_minimal(graph)  # remove extra text
    #
    # # word形式存储
    with open("tree.dot", 'w') as f:
        f = export_graphviz(tree, out_file=f, class_names=target_names,
                            filled=True, rounded=True)
    graph.write_pdf(path)


def prune_duplicate_leaves(mdl):
    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(mdl.tree_, decisions)


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        ##print("Pruned {}".format(index))


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


if __name__ == '__main__':
    train_file_path = "datasets/MissileTree_train.csv"
    test_file_path = "datasets/MissileTree_test.csv"
    # blackbox_file_path = "blackbox/mlp_wine.model"
    features = getFeatureNames(train_file_path)
    target_names = ['0', '1', '2', '3']
    blackbox_type = "RF"

    # 读取数据集
    X_train, y_train = getTrainOrTestData(train_file_path)
    X_test, y_test = getTrainOrTestData(test_file_path)

    ### 项目用
    bb_y_train = y_train
    bb_y_test = y_test
    ###

    # 读取黑盒模型
    # with open(blackbox_file_path, 'rb') as f:
    #     blackbox = joblib.load(f)
    #     if blackbox_type == "MLP":
    #         mms = MinMaxScaler()
    #         # mlp才需要的归一化，因为我对mlp的输入特征做了归一化处理
    #         mms_data_train = mms.fit_transform(X_train)
    #         mms_data_test = mms.fit_transform(X_test)
    #         bb_y_train = blackbox.predict(mms_data_train)
    #         bb_y_test = blackbox.predict(mms_data_test)
    #     else:
    #         bb_y_train = blackbox.predict(X_train)
    #         bb_y_test = blackbox.predict(X_test)
    # 训练决策树
    tree = DecisionTreeClassifier(random_state=1, min_samples_leaf=5, criterion='gini')
    tree = tree.fit(X_train, bb_y_train)
    visualize_decisionTree(tree, 'before_prune.pdf', target_names, features)
    prune_duplicate_leaves(tree)
    visualize_decisionTree(tree, 'after_prune.pdf', target_names, features)
    # 保存决策树
    print("-=================")
    print("node count")
    print(tree.tree_.node_count)
    print("node threshold")
    print(tree.tree_.threshold)
    print("node feature.txt")
    print(tree.tree_.feature)
    print("node children right")
    print(tree.tree_.children_right)
    print("node children left")
    print(tree.tree_.children_left)
    print('feature_names', features)