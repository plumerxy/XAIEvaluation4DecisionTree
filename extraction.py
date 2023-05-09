import joblib
import numpy as np
from causalml.inference.meta import XGBTRegressor
from sklearn.metrics import roc_auc_score
import collections
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pydot
import pandas as pd
import sys
from os import path
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score
import random
import rule_extraction

from causalml.inference.meta import XGBTRegressor, MLPTRegressor

# 先获取当前运行时临时目录路径
if getattr(sys, 'frozen', None):
    basedir = sys._MEIPASS
else:
    basedir = path.dirname(__file__)

"""
树结构：
节点标号从0开始 先序遍历
node_count：int
threshold：numpy.ndarray 
feature.txt：numpy.ndarray
children_left：numpy.ndarray
children_right：numpy.ndarray
label:节点对应的类别 numpy.ndarray
注意：feature.txt and threshold only apply to split nodes.
"""


class Tree_:
    def __init__(self, node_count=0, threshold=None, feature=None, children_left=None, children_right=None, label=None):
        self.node_count = node_count
        self.threshold = threshold
        self.feature = feature
        self.children_left = children_left
        self.children_right = children_right
        self.label = label


class TreeNode:
    def __init__(self, num=0, feature=0, threshold=0, left=None, right=None, up=None, label=None, angle=None,
                 fill_color=None):
        self.num = num
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.up = up
        self.label = label
        self.angle = angle
        self.fill_color = fill_color


"""
构建树 
返回先序节点列表
"""


def buildTree(tree_):
    treenode_list = []
    for i in range(tree_.node_count):
        treenode = TreeNode(i, tree_.feature[i], (tree_.threshold[i]), tree_.children_left[i],
                            tree_.children_right[i], None, tree_.label[i])
        treenode_list.append(treenode)
    for i in range(tree_.node_count):
        treenode = treenode_list[i]
        if treenode.left != -1:
            treenode.left = treenode_list[treenode.left]
            treenode.left.angle = 45
            treenode_list[treenode.left.num].up = treenode
        if treenode.right != -1:
            treenode.right = treenode_list[treenode.right]
            treenode.right.angle = -45
            treenode_list[treenode.right.num].up = treenode
    return treenode_list


'''
读决策树信息文件
返回值包含：
node_count：int
threshold：numpy.ndarray 
feature.txt：numpy.ndarray
children_left：numpy.ndarray
children_right：numpy.ndarray
label:节点对应的类别 numpy.ndarray
'''


def read_dataset(filename):
    fr = open(filename, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    dataset = []
    for line in all_lines[0:6]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        line_ = np.array(list(map(float, line)))
        dataset.append(line_)
    dataset[0] = int(dataset[0])  # node_count转为int 其余为numpy.array
    # 除了threshold的numpy里面存储的类型为float 其余的numpy里面存储的类型都为int
    dataset[2] = dataset[2].astype(int)
    dataset[3] = dataset[3].astype(int)
    dataset[4] = dataset[4].astype(int)
    dataset[5] = dataset[5].astype(int)
    return dataset


# 输入存储树信息的文件路径 返回封装好的tree_结构
def getTree(filepath):
    result = read_dataset(filepath)
    tree_ = Tree_(result[0], result[1], result[2], result[3], result[4], result[5])
    return tree_


# 读决策树信息文件 获得特征文本
def getFeatureText(filepath):
    fr = open(filepath, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    feature_text = []
    for line in all_lines[6:7]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        for word in line:
            feature_text.append(word)
    return feature_text


# 读决策树信息文件 获得分类的类别标签文本
def getLabelText(filepath):
    fr = open(filepath, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    label_text = []
    for line in all_lines[7:8]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        for word in line:
            label_text.append(word)
    return label_text


# 计算APL (训练集上样本的平均决策路径)
def average_path_length(tree, X, class_dict):
    """Compute average path length: cost of simulating the average
    example; this is used in the objective function.

    @param tree: DecisionTreeClassifier instance
    @param X: NumPy array (D x N)
              D := number of dimensions
              N := number of examples
    @return path_length: float
                         average path length
    """
    children_left = tree.children_left
    children_right = tree.children_right
    # leaf_indices = tree.apply(X)
    # leaf_indices = tree.apply(X)#获得样本对应的叶子节点编号序列 https://www.zhihu.com/question/39254529
    # leaf_counts = np.bincount(leaf_indices)#统计上面每个叶子节点编号出现的次数
    node_depth = np.zeros(shape=class_dict.shape, dtype=np.int64)
    is_leaves = np.zeros(shape=class_dict.shape, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        # If we have a test node
        if ((children_left[node_id] != children_right[node_id]) and children_left[node_id] != -1):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    path_length = np.dot(node_depth, class_dict) / float(X.shape[0])
    return path_length


'''
计算节点数量（去除了无效分支（左右子树类别一样）节点）
'''


def countNodeCount(tree):
    # 与children_left[i], children_right[i]相关的建议仍旧使用自带的node_count属性
    count = 1
    for i in tree.children_right:
        if i != -1:
            count += 1
    for i in tree.children_left:
        if i != -1:
            count += 1
    return count


def collect(node, counter):
    if node == -1 or (node.left == -1 and node.right == -1):
        return "#"
    serial = "{},{},{}".format(node.feature, collect(node.left, counter),
                               collect(node.right, counter))
    counter[serial] += 1
    return serial


# 计算重复子树
def countDuplicateSubstree(tree):
    # print("start countDuplicateSubstree")
    counter = collections.Counter()  # 计数器
    duplicate_subtree = 0
    n_nodes = int(tree.node_count)
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    label = tree.label

    # 利用children_left和children_right构造TreeNode形式的树
    treenode_list = []
    for i in range(n_nodes):
        treenode = TreeNode(i, feature[i], threshold[i], children_left[i], children_right[i], None, label[i])
        treenode_list.append(treenode)
    for i in range(n_nodes):
        treenode = treenode_list[i]
        if treenode.left != -1:
            treenode.left = treenode_list[treenode.left]
        if treenode.right != -1:
            treenode.right = treenode_list[treenode.right]

    # 将所有子树以[属性，阈值，左子树，右子树]进行序列化，在counter中统计每种序列化形式出现的次数
    collect(treenode_list[0], counter)

    # 出现次数大于等于2即表示有重复子树
    for item in counter:
        if counter[item] >= 2:
            duplicate_subtree += len(item.split(","))
            # duplicate_subtree += len(item.split(","))*counter[item]
    return duplicate_subtree * 2 / countNodeCount(tree)


'''
计算重复属性
'''


def countAverageDuplicateAttr(tree, X, class_dict):
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    label = tree.label
    threshold = tree.threshold

    # 利用children_left和children_right构造TreeNode形式的树
    treenode_list = []
    for i in range(n_nodes):
        treenode = TreeNode(i, feature[i], threshold[i], children_left[i], children_right[i], None, label[i])
        treenode_list.append(treenode)
    for i in range(n_nodes):
        treenode = treenode_list[i]
        if treenode.left != -1:
            treenode.left = treenode_list[treenode.left]
            treenode_list[treenode.left.num].up = treenode
        if treenode.right != -1:
            treenode.right = treenode_list[treenode.right]
            treenode_list[treenode.right.num].up = treenode

    duplicate_attr_list = []
    for i in range(class_dict.size):
        feature_dict = {}
        duplicate_attr = 0
        node_in_path = 0
        if class_dict[i] != 0:
            t = treenode_list[i]
            while (t.up != None):
                feature_dict[t.up.feature] = feature_dict.setdefault(t.up.feature, 0) + 1
                t = t.up
                node_in_path += 1
            for key, value in feature_dict.items():
                if value >= 2 and key != -2:
                    duplicate_attr += value

        if node_in_path != 0:
            duplicate_attr_list.append(duplicate_attr / node_in_path)
        else:
            duplicate_attr_list.append(0.0)
    path_length = np.dot(duplicate_attr_list, class_dict) / float(X.shape[0])
    return path_length


'''
treenode:根节点
sample:样本
调用时
class_dict!=None class_dict获取落在每个节点上的样本数量 key:叶子节点编号 value:落在每个叶子节点上的样本数量
sample_labels！=None test_labels存储数据集sample中样本对应的标签列表
leave_dict key:叶子节点编号 value:对应样本编号列表 key:叶子节点编号 value:落在该叶子节点上的样本编号列表
sample_label_dict:key:叶子节点编号 value:数据集落在该叶子节点的标签列表
leave_node_list:样本对应的叶子节点列表
leave_node_set:获得叶子节点集合
class_num_list:获得每类的样本数量
black_label:黑盒预测结果
blackbox_label_dict:黑盒预测的结果  key:叶子节点编号 value：该叶子节点上黑盒预测结果列表
'''


def getSampleClass(treenode, sample, class_dict, sample_labels, leave_dict, sample_label_dict, leave_node_list,
                   leave_node_set, class_num_list, black_label, blacklabels_dict):
    if (treenode == -1):
        return
    featurn_index = treenode.feature
    if (sample[featurn_index] <= treenode.threshold):
        getSampleClass(treenode.left, sample, class_dict, sample_labels, leave_dict, sample_label_dict, leave_node_list,
                       leave_node_set, class_num_list, black_label, blacklabels_dict)
    else:
        getSampleClass(treenode.right, sample, class_dict, sample_labels, leave_dict, sample_label_dict,
                       leave_node_list, leave_node_set, class_num_list, black_label, blacklabels_dict)
    if (treenode.left == -1 and treenode.right == -1):
        if (class_dict != None):
            class_dict[treenode.num] = class_dict[treenode.num] + 1
        if (sample_labels != None):
            sample_labels.append(treenode.label)
        if (leave_dict != None):
            if (sample_label_dict.get(treenode.num) == None):
                sample_label_dict[treenode.num] = [treenode.label]
                blacklabels_dict[treenode.num] = [black_label]
            else:
                sample_label_list = sample_label_dict.get(treenode.num)
                sample_label_list.append(treenode.label)
                sample_label_dict[treenode.num] = sample_label_list
                black_label_list = blacklabels_dict.get(treenode.num)
                black_label_list.append(black_label)
                blacklabels_dict[treenode.num] = black_label_list
            if (leave_dict.get(treenode.num) == None):
                leave_dict[treenode.num] = [sample]
            else:
                sampel_List = leave_dict.get(treenode.num)
                sampel_List.append(sample)
                leave_dict[treenode.num] = sampel_List
        if (leave_node_list != None):
            leave_node_list.append(treenode)
        if (leave_node_set != None):
            leave_node_set.add(treenode)
        if (class_num_list != None):
            class_num_list[treenode.label] += 1


def getSamplesClass(treenode, samples, class_dict, test_labels, leave_dict, sample_label_dict, leave_node_list,
                    leave_node_set, class_num_list, blackbox_labels, blacklabels_dict):
    samples_list = samples.tolist()
    blackbox_label = None
    for i in range(len(samples_list)):
        sample = samples_list[i]
        if (blacklabels_dict != None):
            blackbox_label = blackbox_labels[i]
        getSampleClass(treenode, sample, class_dict, test_labels, leave_dict, sample_label_dict, leave_node_list,
                       leave_node_set, class_num_list, blackbox_label, blacklabels_dict)
    return class_dict


'''
计算因果性
'''


def getCausal(X_train, Y_train, threshold, feature, feature_name):
    df = pd.DataFrame(X_train, columns=feature_name)
    y = Y_train
    ate = []
    for i, fe in enumerate(feature):
        if fe != -2:
            treatment = np.array([(1 if val <= threshold[i] else 0) for val in df.iloc[:, fe].values]).astype(int)
            X = df.drop(df.columns[fe], axis=1).values
            xg = XGBTRegressor(random_state=42)
            te, lb, ub = xg.estimate_ate(X, treatment, y)
            ate.append(te)
    return np.mean(ate)


'''
可视化决策树
'''


def visualize_decisionTree(tree, path, target_names, features):
    dot_data = export_graphviz(tree, out_file=None, class_names=target_names,
                               filled=True, rounded=True, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # # graph = make_graph_minimal(graph)  # remove extra text
    # # word形式存储
    with open("proxy/CART_mlp_breast/tree.dot", 'w') as f:
        f = export_graphviz(tree, out_file=f, class_names=["maligent", "begine"],
                            filled=True, rounded=True)
    graph.write_pdf(path)


'''
 计算特征范围，返回值为最小值与最大值
 array:数据集
 index:计算指定下标特征的最小值与最大值
'''


def getAttributeRange(array, index):
    tup = array.shape
    row = tup[0]
    max = array[0][index]
    min = array[0][index]
    for i in range(int(row)):
        if max < array[i][index]:
            max = array[i][index]
        if min > array[i][index]:
            min = array[i][index]
    return min, max


'''
计算两棵树之间的不相似性（结构稳定性）
'''


def getDisimilarity(tree1, tree2, data):
    n_nodes = tree1.tree_.node_count
    n_splitnode = n_nodes - tree1.tree_.n_leaves
    children_left = tree1.tree_.children_left
    children_right = tree1.tree_.children_right
    feature = tree1.tree_.feature
    threshold = tree1.tree_.threshold
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    n_nodes1 = tree2.tree_.node_count
    n_splitnode1 = n_nodes1 - tree2.tree_.n_leaves
    children_left1 = tree2.tree_.children_left
    children_right1 = tree2.tree_.children_right
    feature1 = tree2.tree_.feature
    threshold1 = tree2.tree_.threshold
    node_depth1 = np.zeros(shape=n_nodes1, dtype=np.int64)
    stack1 = [(0, 0)]
    sum_similarity = 0
    dis = 0
    while len(stack) > 0 and len(stack1) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        node_id1, depth1 = stack1.pop()
        node_depth1[node_id1] = depth1
        is_split_node = children_left[node_id] != children_right[node_id]
        is_split_node1 = children_left1[node_id1] != children_right1[node_id1]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))

        if is_split_node1:
            stack1.append((children_left1[node_id1], depth1 + 1))
            stack1.append((children_right1[node_id1], depth1 + 1))

        if is_split_node1 and is_split_node:  # 计算不为0的s

            if feature1[node_id1] == feature[node_id]:
                weight = min(n_splitnode1, n_splitnode)
                minValue, maxValue = getAttributeRange(data, feature[node_id])
                fanwei = maxValue - minValue
                sim = (1 - (abs(threshold[node_id] - threshold1[node_id1])) / fanwei)
                sum_similarity += sim * (1 / weight)
    dis += (1 - (sum_similarity))
    return dis


'''
计算相似性（语义、结构）
split_number：交叉验证的折数
data：训练集
labels:训练集对应标签
blackbox_model：黑盒模型
surrogate_model：代理模型
tree_save_path：训练集交叉验证产生不同决策树的图像保存地址
X_test：测试集数据集
number：语义稳定性 测试集取样数量
target_names：类名称列表
features:特征列表
'''


def getSimilarity(split_number, data, labels, blackbox_model, surrogate_model, tree_save_path, X_test, number,
                  target_names, features):
    # 十折交叉验证 生成十颗决策树 用于计算不稳定性
    tree_list = []
    KF = KFold(n_splits=split_number, random_state=None, shuffle=False)
    index_number = 0
    # 计算语义稳定性
    test_labels = []
    samples = copy.deepcopy(X_test)
    np.random.shuffle(samples)
    # 随机测试集取样本
    samples = samples[0:number]
    for train_index, test_index in KF.split(data):
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = labels[train_index], labels[test_index]
        blackbox_train_label = blackbox_model.predict(X_train)
        surrogate_model_new = copy.deepcopy(surrogate_model)
        surrogate_model_new.fit(X_train, blackbox_train_label)
        test_labels.append(surrogate_model_new.predict(samples))
        visualize_decisionTree(surrogate_model_new, tree_save_path + str(index_number) + ".pdf", target_names, features)
        # visualize_decisionTree(tree_fold, tree_fold_path)
        tree_list.append(surrogate_model_new)
        index_number += 1

    # 计算不稳定性
    sum_dis = 0
    sum_semantic_similarity = 0
    for i in range(split_number):
        j = i + 1
        while j < split_number:
            dis = getDisimilarity(tree_list[i], tree_list[j], data)
            sum_semantic_similarity += accuracy_score(test_labels[i], test_labels[j])
            sum_dis += dis
            j += 1
    result_structural = round(1 - sum_dis / (split_number * (split_number - 1) / 2), 3)
    result_semantic = sum_semantic_similarity / ((split_number * (split_number - 1)) / 2)
    return result_structural, result_semantic


'''
干扰样本，干扰样本的全部特征
range:干扰范围 若干扰特征取值范围的5% 即填0.05
sample：需要干扰的样本
返回干扰后的样本
'''


def disturbSample(range, sample):
    disturbSample = copy.deepcopy(sample)
    feature_min_list = []
    feature_max_list = []
    feature_number = disturbSample.shape[1]
    sample_number = disturbSample.shape[0]
    # 获取对应数据集上全部特征的取值范围 存储到feature_range_list
    i = 0
    j = 0
    index = 0
    while i < feature_number:
        feature_min_list.append(getAttributeRange(disturbSample, i)[0])
        feature_max_list.append(getAttributeRange(disturbSample, i)[1])
        i += 1
    while j < sample_number:
        while index < feature_number:
            fanwei = feature_max_list[index] - feature_min_list[index]
            disturbSample[j][index] = random.uniform(disturbSample[j][index] - range * fanwei,
                                                     disturbSample[j][index] + range * fanwei)
            index += 1
        j += 1
    return disturbSample


'''
根据叶子节点获得路径使用的特征
leave_node_list：存储样本对应的叶子节点
返回result列表 存储每个样本使用的特征set集合
'''


def getPathByLeaveNode(leave_node_list):
    result = []
    for i in range(len(leave_node_list)):
        treeNode = leave_node_list[i].up
        path_set = set()
        while treeNode != None:
            path_set.add(treeNode.feature)
            treeNode = treeNode.up
        result.append(path_set)
    return result


'''
计算稳定性：相似输入有相似解释
干扰样本全部特征范围的5% 计算干扰前后的相似性
相似性：分子：解释采用特征的交集 分母：解释采用特征的并集
返回的全部样本干扰前后相似性的均值
'''


def getSimilarityOfInter(sample, disturbRange, treeNode):
    disturb = disturbSample(disturbRange, sample)
    # 获取样本对应叶子节点列表
    leave_node_list = []
    getSamplesClass(treeNode, sample, None, None, None, None, leave_node_list, None, None, None, None)
    path_list = getPathByLeaveNode(leave_node_list)
    leave_node_disturb_list = []
    getSamplesClass(treeNode, disturb, None, None, None, None, leave_node_disturb_list, None, None, None, None)
    path_disturb_list = getPathByLeaveNode(leave_node_disturb_list)
    sum_similarity_Inter = 0
    for i in range(len(path_list)):
        similarity_Inter = len(path_list[i] & path_disturb_list[i]) / len(path_list[i] | path_disturb_list[i])
        sum_similarity_Inter += similarity_Inter
    return sum_similarity_Inter / (len(path_list))


'''
抽取样本
percentage:抽样比例 用小数表示 抽10% 即填0.1
sample_dict：key：叶子节点编号 value:落在此叶子节点上的数据集列表（不含标签）
sample_label_dict：key:叶子节点编号 value:落在此叶子节点与上面数据集对应的标签列表
抽样后直接修改sample_label_dict值为对应抽样样本的标签
blackbox_test_label: key:叶子节点编号 value:落在叶子节点上黑盒对应的标签列表
抽样后直接修改blackbox_test_label值为对应抽样样本的标签
'''


def getSmaple(percentage, sample_dict, sample_label_dict, blackbox_test_label):
    # 存储最终选取的样本
    choice_dict = {}
    # key:叶子节点的编号 value:存储样本下标编号的列表
    for key in sample_dict.keys():
        sample_dict[key] = np.array(sample_dict.get(key))
        if (int(sample_dict.get(key).shape[0] * percentage)) < 10:
            number = min(sample_dict.get(key).shape[0], 10)
        else:
            number = int(sample_dict.get(key).shape[0] * percentage)
        # np.random.shuffle(sample_dict[key])
        choice_dict[key] = sample_dict[key][0:number]
        sample_label_dict[key] = sample_label_dict[key][0:number]
        blackbox_test_label[key] = blackbox_test_label[key][0:number]
    return choice_dict


'''
获取每条路径选取的特征编号
从每个叶子节点向上跳，直到跳到根节点
leave_dict：key：叶子节点编号 value:落在该叶子节点的数据集列表（不含标签）
tree_list：先序遍历存储树的每个节点（treenode结构）
path_dict: key:叶子节点编号 value: 对应路径特征编号列表
'''


def getPath(leave_dict, tree_list):
    path_dict = {}
    for key in leave_dict.keys():
        treenode = tree_list[key]
        feature_list = []
        treenode = treenode.up
        while treenode != None:
            feature_list.append(treenode.feature)
            treenode = treenode.up
        path_dict[key] = feature_list
    return path_dict


'''
遮挡无用特征
choice_dict:抽样选取的样本 key:叶子节点编号 value:落在该叶子节点的数据集列表（不含标签）
feature_dict：key:叶子节点编号 value:该条路径选取分支节点对应的特征编号列表
value:遮挡值
'''


def delUnusedFeatures(choice_dict, feature_dict, value, feature_num):
    res_dict = copy.deepcopy(choice_dict)
    for key in res_dict.keys():
        for i in range(len(res_dict.get(key))):
            for j in range(feature_num):
                if j not in feature_dict.get(key):
                    res_dict.get(key)[i][j] = value
    return res_dict


'''
有效性
遮挡样本决策未使用的特征 让黑盒预测 计算黑盒遮挡前后预测的accuracy
suff_dict:遮挡后的样本 key:叶子节点编号 value:落在该叶子节点上的数据集列表（遮挡了无关特征 不含标签）
sample_label_dict：key:叶子节点编号 value:对应上面样本的标签列表
blackbox_label_dict:key:叶子节点编号 value:对应叶子节点编号黑盒标签列表
'''


def getSufficiency(suff_dict, sample_label_dict, blackbox_label_dict):
    res = 0
    size = 0
    for k in sample_label_dict.keys():
        size += len(sample_label_dict.get(k))
    for key in suff_dict.keys():
        same_count = 0
        sample_num = len(sample_label_dict.get(key))
        for i in range(sample_num):
            if (blackbox_label_dict[key][i] == sample_label_dict[key][i]):
                same_count += 1
        res += (sample_num / size) * (same_count / sample_num)
    return res


''''
输入叶子节点列表
从叶子节点开始获得特征编号列表
path_feature_dict：key:叶子节点编号 value：特征列表（从叶子节点开始）
'''


def getPathFeatureList(leave_node_list):
    path_feature_dict = {}
    for node in (leave_node_list):
        path_feature_list = [node.num]
        treenode = node.up
        while treenode != None:
            path_feature_list.append(treenode.num)
            treenode = treenode.up
        path_feature_dict[node.num] = path_feature_list
    return path_feature_dict


'''
根据叶子节点获得路径使用的特征及阈值
leave_node_list：存储样本对应的叶子节点
返回result列表 Key：叶子节点编号 value {key:feature value:threshold}
feature=-2 存储的为类别
'''


def getPathFeatureAndThreshold(leave_node_list, tree_list):
    path_feature_dict = getPathFeatureList(leave_node_list)
    result = {}
    for node in leave_node_list:
        path_map = {}
        path_map[-2] = [node.label]
        treeNode = node.up
        while treeNode != None:
            feature_list = path_feature_dict[node.num]
            node_num = feature_list[feature_list.index(treeNode.num) - 1]
            angle = tree_list[node_num].angle
            if (path_map.get(treeNode.feature) == None):
                path_map[treeNode.feature] = [[treeNode.threshold, angle]]
            else:
                path_list = path_map[treeNode.feature]
                path_list.append([treeNode.threshold, angle])
                path_map[treeNode.feature] = path_list
            treeNode = treeNode.up
        result[node.num] = path_map
    return result


'''
获取最多类的下标
class_num_list:每类样本个数列表
返回个数最多的类下标
'''


def getMostNumClass(class_num_list):
    index = -1
    max_value = -1
    for i in range(len(class_num_list)):
        if (class_num_list[i] > max_value):
            max_value = class_num_list[i]
            index = i
    return index


'''
转换为规则时 处理重复属性
feature：特征名称列表
threshold_list：[阈值，角度] 45为左箭头，对应True -45为右箭头，对应False
'''


def processDuplicateItem(feature, threshold_list):
    # 默认画图分支节点为<=
    # 45 对应 <=
    min_value = float('inf')
    # -45 对应 >
    max_value = float('-inf')
    for li in threshold_list:
        if li[1] == 45:
            min_value = min(min_value, li[0])
        else:
            max_value = max(max_value, li[0])
    if min_value != float('inf') and max_value != float('-inf'):
        return '\'' + str(max_value) + ' < ' + feature + ' <= ' + str(min_value) + '\','
    elif min_value != float('inf'):
        return '\'' + feature + ' <= ' + str(min_value) + '\','
    elif max_value != float('-inf'):
        return '\'' + feature + ' > ' + str(max_value) + '\','


'''
计算有效性
将决策树转换为指定规则格式
'''


def getEffectiveness(leave_list, feature_text, tree_list, X_train, treenode, class_num_list):
    result = getPathFeatureAndThreshold(leave_list, tree_list)
    # getSamplesClass(treenode,X_train,None,None,None,None,None,None,class_num_list,None,None)
    most_class = getMostNumClass(class_num_list)
    rule = ''
    for path in result:
        rule += '('
        for feature in result[path]:
            if feature != -2:
                feature_list = result[path].get(feature)
                if len(feature_list) > 1:
                    rule += processDuplicateItem(feature_text[feature], feature_list)
                else:
                    for threshold_list in feature_list:
                        if threshold_list[1] == 45:
                            sign = ' <= '
                        else:
                            sign = ' > '
                        rule += '\'' + feature_text[feature] + sign
                        rule += str(threshold_list[0]) + '\','
            else:
                label = result[path][-2][0]
        rule = rule[0:-1]
        rule += '),' + str(label) + '.0\n'
    rule += str(most_class) + '.0'
    return rule


'''
计算有效性
将决策树转换为指定规则格式
'''


def getEffectivenessMulti(leave_list, feature_text, tree_list, X_train, treenode, class_num_list, class_number):
    result = getPathFeatureAndThreshold(leave_list, tree_list)
    # getSamplesClass(treenode,X_train,None,None,None,None,None,None,class_num_list,None,None)
    most_class = getMostNumClass(class_num_list)
    rule = ''
    label_list = [0.0] * class_number
    for path in result:
        label_list_tmp = copy.deepcopy(label_list)
        rule += '('
        for feature in result[path]:
            if feature != -2:
                feature_list = result[path].get(feature)
                if len(feature_list) > 1:
                    rule += processDuplicateItem(feature_text[feature], feature_list)
                else:
                    for threshold_list in feature_list:
                        if threshold_list[1] == 45:
                            sign = ' <= '
                        else:
                            sign = ' > '
                        rule += '\'' + feature_text[feature] + sign
                        rule += str(threshold_list[0]) + '\','
            else:
                label = result[path][-2][0]
                label_list_tmp[label] = 1.0
        rule = rule[0:-1]
        rule += '),' + str(label_list_tmp).replace('[', '').replace(']', '') + '\n'
    label_list_tmp = copy.deepcopy(label_list)
    label_list_tmp[most_class] = 1.0
    rule += str(label_list_tmp).replace('[', '').replace(']', '')
    print(rule)
    return rule


'''
提取决策树的指标 写入index_path
指标提取 除了稳定性指标是十折交叉验证取平均值外 其余指标都通过上面生成的决策树进行计算
tree:树结构的决策树对应的根节点
X_train:x训练集
X_test:测试集
class_dict:落在每个节点的样本数量
base_path:指标存储路径、规则存储路径前缀
blackbox_model:黑盒模型
surrogate_model:代理模型
suff_percentage:充分性取样比例
sim_percentage:稳定性（相似输入有相似解释）取样比例
disturbRange:稳定性（相似输入有相似解释）干扰范围
sim_pic_basepath:交叉验证生成不同决策树图片的存储路径
feature_text:特征文本列表
y_train:训练集标签
target_names:类名列表
features：特征列表
class_num_list: 每类样本个数列表
shelter_value:特征遮挡值
'''


def getDecisionTreeIndexes(tree_, X_train, X_test, class_dict, base_path, blackbox_model, surrogate_model,
                           suff_percentage, sim_percentage, disturbRange, sim_pic_basepath, split_number,
                           semantic_sim_number, feature_text, tree_list, y_train, target_names, features,
                           class_num_list, shelter_value, mms, black_type):
    result = {}
    treeNode = tree_list[0]
    # 一致性
    if black_type == "MLP":
        mms_test = mms.transform(X_test)
    else:
        mms_test = X_test
    blackbox_test_label = blackbox_model.predict(mms_test)
    # tree_test_label=surrogate_model.predict(X_test)
    # 存储决策树训练集标签
    tree_test_label = []
    # 获取落在每个叶子节点上的样本和对应黑盒的标签
    leave_dict = {}
    sample_label_dict = {}
    black_label_dict = {}
    getSamplesClass(treeNode, X_test, None, tree_test_label, leave_dict, sample_label_dict, None, None, None,
                    blackbox_test_label, black_label_dict)
    label_size = len(set(list(blackbox_test_label)))
    if label_size == 2:
        AUC = roc_auc_score(blackbox_test_label, tree_test_label)
    else:
        enc = OneHotEncoder()
        enc.fit(np.array((list(set(list(blackbox_test_label))|set(tree_test_label)))).reshape(-1, 1))
        y_true = enc.transform(blackbox_test_label.reshape(-1, 1)).toarray()
        y_pred = enc.transform(np.array(tree_test_label).reshape(-1, 1)).toarray()
        AUC = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')

    result['AUC'] = round(AUC, 3)

    # 复杂性
    APL = average_path_length(tree_, X_train, class_dict)
    result['APL'] = round(APL, 3)
    node_count = countNodeCount(tree_)
    result['node count'] = node_count

    # 明确性
    duplicate_subtree = countDuplicateSubstree(tree_)
    result['duplicate_subtree'] = duplicate_subtree
    duplicate_attr = countAverageDuplicateAttr(tree_, X_train, class_dict)
    result['duplicate_attr'] = duplicate_attr

    # 稳定性（结构稳定性、语义稳定性）
    # structural_similarity,semantic_similarity=getSimilarity(split_number,X_train,y_train,blackbox_model,surrogate_model,sim_pic_basepath,X_test,semantic_sim_number,target_names,features)
    # result['structural_similarity']=structural_similarity
    # result['semantic_similarity']=semantic_similarity

    # 稳定性 相似输入有相似解释
    similarity_sample = X_test
    np.random.shuffle(similarity_sample)
    similarity_sample = similarity_sample[0:int(similarity_sample.shape[0] * sim_percentage)]
    inter_similarity = getSimilarityOfInter(similarity_sample, disturbRange, treeNode)
    result['inter_similarity'] = inter_similarity

    # # 充分性
    choice_dict = getSmaple(suff_percentage, leave_dict, sample_label_dict, black_label_dict)
    feature_dict = getPath(leave_dict, tree_list)
    suff_dict = delUnusedFeatures(choice_dict, feature_dict, shelter_value, X_train.shape[1])
    sufficiency = getSufficiency(suff_dict, sample_label_dict, black_label_dict)
    result['sufficiency'] = sufficiency

    # # 因果性
    ATE = getCausal(X_train, y_train, tree_.threshold, tree_.feature, features)
    result['ATE'] = ATE

    # with open(base_path+'/index.txt', "w+") as f:
    #     f.write(str(result))
    return result



# 递归存储每个节点用于画图的文本
def visualizeCore(treenode, list, feature_text, label_text, color_list):
    if (treenode == -1):
        return
    # 叶子节点不显示特征和阈值
    if (treenode.left == -1 and treenode.right == -1):
        list.append(
            str(treenode.num) + ' [label="\\nclass=' + label_text[treenode.label] + '", fillcolor=' + '"' + color_list[
                treenode.label] + '"];')
    # 分支节点显示特征和阈值
    else:
        list.append(str(treenode.num) + ' [label="' + (feature_text[treenode.feature]) + '<=' + str(
            treenode.threshold) + '", fillcolor=' + '"' + color_list[treenode.label] + '"];')
    # 有父节点的 创建与父节点的连线
    if (treenode.up != None):
        if (treenode.up.num == 0 and treenode.up.left.num == treenode.num):
            list.append(
                '\n' + str(treenode.up.num) + ' -> ' + str(treenode.num) + ' [labeldistance=2.5, labelangle=' + str(
                    treenode.angle) + ', headlabel="True"];')
        elif (treenode.up.num == 0 and treenode.up.right.num == treenode.num):
            list.append(
                '\n' + str(treenode.up.num) + ' -> ' + str(treenode.num) + ' [labeldistance=2.5, labelangle=' + str(
                    treenode.angle) + ', headlabel="False"];')
        else:
            list.append(
                '\n' + str(treenode.up.num) + ' -> ' + str(treenode.num) + ' [labeldistance=2.5, labelangle=' + str(
                    treenode.angle) + '];')
    list.append('\n')
    visualizeCore(treenode.left, list, feature_text, label_text)
    visualizeCore(treenode.right, list, feature_text, label_text)


'''
生成决策树信息文件对应决策树的画图文本
treenode:根节点
feature_text：特征名称列表
label_text：列表名称列表
tree_dot_path：dot文件存储路径
list:存储dot文本信息
'''


def visualize(treenode, feature_text, label_text, tree_dot_path, color_list):
    list = []
    list.append(
        'digraph Tree {node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;edge [fontname=helvetica] ;' + '\n')
    visualizeCore(treenode, list, feature_text, label_text, color_list)
    list.append('\n' + '}')
    with open(tree_dot_path, "w+") as f:
        f.write(''.join(list))


# 生成决策树png图片
# doc_path：用于画决策树的dot文件路径
# pic_path：生成决策树png图像的路径
def showTree(doc_path, pic_path):
    (graph,) = pydot.graph_from_dot_file(doc_path)
    graph.write_png(pic_path);


if __name__ == '__main__':
    # 参数
    # 交叉验证折数
    split_number = 5
    # 稳定性（相似输入相似解释）干扰范围 10%即填0.1
    disturbRange = 0.1
    # 稳定性（相似输入相似解释） 取样比例 10%即0.1
    sim_percentage = 0.1
    # 语义稳定性 测试集取样数量
    semantic_sim_number = 100
    # 充分性取样比例 10%即填0.1
    suff_percentage = 0.1
    # 代理模型路径
    tree_model_path = "tree.pkl"
    # 指标存储路径
    base_path = 'result'
    # 交叉验证生成不同决策树图片存储路径前缀
    sim_pic_basepath = "similarity/"
    # 特征遮挡值
    shelter_value = 6
    # 文件路径
    dataset_train_path = "datasets/breastcancer_train.csv"
    dataset_test_path = "datasets/breastcancer_test.csv"
    blackbox_path = "blackbox/rf_breast.model"
    proxy_file_path = "proxy/CHAID_rf_breast.txt"
    blackbox_type = "RF"

    # 加载数据集
    """从dataset_test加载样本、特征列表"""
    csv_data = pd.read_csv(dataset_train_path, low_memory=False)  # 防止弹出警告
    feature_name = csv_data.columns[:-1]
    X_train = np.array(csv_data)[:, :-1]
    y_train = np.array(csv_data)[:, -1]
    csv_data = pd.read_csv(dataset_test_path, low_memory=False)  # 防止弹出警告
    X_test = np.array(csv_data)[:, :-1]
    y_test = np.array(csv_data)[:, -1]

    # """判断分类任务类型"""
    # label_size = len(set(list(y_train)))
    # target_names = []
    # class_num_list = []
    # for i in range(label_size):
    #     # 类名称列表
    #     target_names.append('Class'+str(i))
    #     # 统计每类样本个数 有几类列表中有几个0
    #     class_num_list.append(0)

    # """for case study"""
    # blackbox = joblib.load(blackbox_path)
    # mms = MinMaxScaler()
    # X_train = mms.fit_transform(X_train)
    # X_test = mms.fit_transform(X_test)
    # # 读取决策树文件
    # decision_info_path = proxy_file_path  # 决策树信息文件存储地址
    # tree_ = getTree(decision_info_path)
    # ATE = getCausal(X_train, y_train, tree_.threshold, tree_.feature, feature_name)
    # print("ate: %.4f" %ATE)

    """提取指标"""
    """ 加载黑盒、决策树 """
    blackbox = joblib.load(blackbox_path)
    mms = MinMaxScaler()
    if blackbox_type == "RF":
        # mlp才需要的归一化，因为我对mlp的输入特征做了归一化处理
        mms_data_train = mms.fit_transform(X_train)
        mms_data_test = mms.fit_transform(X_test)
        y_test = blackbox.predict(mms_data_test)
        y_train = blackbox.predict(mms_data_train)
    else:
        y_test = blackbox.predict(X_test)
    # 读取决策树文件
    decision_info_path = proxy_file_path  # 决策树信息文件存储地址
    tree_ = getTree(decision_info_path)
    feature_text = getFeatureText(decision_info_path)  # 获取特征文本（画图）
    target_names = getLabelText(decision_info_path)  # 获取标签文本（画图）
    class_number = len(target_names)
    class_num_list = [0] * class_number

    # 构建树结构 返回根节点
    tree_list = buildTree(tree_)
    treeNode = tree_list[0]
    # # 输出决策树画图用的dot文件
    # visualize_decisionTree(tree_, "tree.pdf", ['0', '1'], feature_text)
    # tree_dot_path = 't.dot'  # 决策树用于画图的dot文件
    # visualize(treeNode, feature_text, lable_text, tree_dot_path)
    # getSamplesClass 存储落在叶子节点上的样本数量 类似之前使用tree 下面两句的功能
    # leaf_indices = tree.apply(X)  # 获得样本对应的叶子节点编号序列 https://www.zhihu.com/question/39254529
    # leaf_counts = np.bincount(leaf_indices)  # 统计上面每个叶子节点编号出现的次数
    class_dict = [0] * tree_.node_count
    leave_node_list = []
    class_dict = getSamplesClass(treeNode, X_train, class_dict, None, None, None, leave_node_list, None,
                                 class_num_list, None, None)
    # 存储测试集决策树预测的结果
    # test_labels = []
    # getSamplesClass(treeNode, X_test, None, test_labels)
    # test_labels = np.array(list(map(int, test_labels)))
    class_dict = np.array(list(map(int, class_dict)))
    # 提取指标
    index_dict = getDecisionTreeIndexes(tree_, X_train, X_test, class_dict, base_path, blackbox,
                                             None, suff_percentage, sim_percentage, disturbRange,
                                             sim_pic_basepath, split_number, semantic_sim_number, feature_text,
                                             tree_list, y_train, target_names, feature_name, class_num_list,
                                             shelter_value, mms, blackbox_type)
    print(index_dict)
