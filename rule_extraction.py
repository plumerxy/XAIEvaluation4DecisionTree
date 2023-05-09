import re
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import joblib
import sys
import random
from os import path
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# 先获取当前运行时临时目录路径
if getattr(sys, 'frozen', None):
    basedir = sys._MEIPASS
else:
    basedir = path.dirname(__file__)

"""*****************************************二分类用代码**************************************************"""


def operation_extract(rule_item):
    # 提取>,<,=,>=,<=
    # rule_item为一个规则项，str
    # operation为要返回的运算符 list
    operation = []
    i = 0
    while i < len(rule_item):
        if rule_item[i] == '=':
            operation.append('=')
            i += 1
        elif rule_item[i] == '>':
            if rule_item[i + 1] == '=':
                operation.append('>=')
                i += 2
            else:
                operation.append('>')
                i += 1
        elif rule_item[i] == '<':
            if rule_item[i + 1] == '=':
                operation.append('<=')
                i += 2
            else:
                operation.append('<')
                i += 1
        else:
            i += 1
    return operation


def feature_name_extract(rule_item, feature_list):
    # 提取属性名
    # rule_item为一个规则项，str
    # str为要返回的运算符 str
    # print(feature_list)
    for feature in feature_list:
        feature = feature + ' '
        if feature in rule_item:
            return feature.strip()
    return None


def compare_operation(op_str, value1, value2):
    # op_str：比较运算符 str
    # value1：样本中的属性值
    # value2：规则的属性值
    # 返回0表示不匹配，1表示匹配
    if op_str == '=':
        if value1 != value2:
            return 0
    elif op_str == '>':
        if value1 <= value2:
            return 0
    elif op_str == '<':
        if value1 >= value2:
            return 0
    elif op_str == '<=':
        if value1 > value2:
            return 0
    else:  # >=
        if value1 < value2:
            return 0
    return 1


def match_onerule_conti(rule, sample, feature_list):
    # rule是从文本中提取的一行规则
    # sample是字典列表
    rule_items = re.findall("'([^']*)'", rule)  # 提取规则项
    if rule_items:
        for item in rule_items:
            operation = operation_extract(item)  # 提取比较运算符
            feature = feature_name_extract(item, feature_list)  # 提取属性名
            if feature == None:
                print("提取特征错误")
            values = re.findall("(\d+\.?\d*)", item)  # 提取属性值（最后一项为概率值）
            # print(operation)
            # print(values)
            n_values =[]
            for x in values:
                if '.' in x:
                    n_values.append(x)
            values = n_values #筛选掉特征中可能包含的数字
            if len(operation) == 1:
                if compare_operation(operation[0], sample[feature], float(values[0])) == 0:
                    return 0
            if len(operation) == 2:
                if ' or ' in item:
                    temp1 = compare_operation(operation[0], sample[feature], float(values[0]))
                    temp2 = compare_operation(operation[1], sample[feature], float(values[1]))
                    if (temp1 or temp2) == 0:
                        return 0
                else:  # and
                    temp1 = compare_operation(operation[0], float(values[0]), sample[feature])
                    temp2 = compare_operation(operation[1], sample[feature], float(values[1]))
                    if (temp1 and temp2) == 0:
                        return 0
        return 1
    else:
        return 0

def get_match_rule(rulelist,sample,feature_list):
    for rule in rulelist:
        if match_onerule_conti(rule,sample,feature_list):
            return rule
    return

def match_rules(rule_list, sample, prob_list, feature_list):
    n = 0
    n_conflict = 0
    class_ = 0
    class_temp = 0
    for i in range(len(rule_list)):
        if match_onerule_conti(rule_list[i], sample, feature_list):
            n += 1
            if i <= len(prob_list) - 1:
                if prob_list[i] > 0.5:
                    class_temp = 1
                else:
                    class_temp = 0
            if n > 1:  # 匹配多个规则
                if class_temp != class_:
                    n_conflict += 1
            class_ = class_temp
    return n, n_conflict  # 匹配的规则数，矛盾的次数


def pred(x_dict, rule_list, feature_list):
    """预测结果计算"""
    for i, rule in enumerate(rule_list):
        if match_onerule_conti(rule, x_dict, feature_list):  # 第一条匹配的规则
            if float(re.search("\),(\d+\.?\d*)", rule).group(1)) > 0.5:
                return 1
            else:
                return 0
    if float(rule_list[-1]) > 0.5:
        return 1
    else:
        return 0


def definiteness(data_list, rule_list, prob, type, feature_list):
    """计算明确性指标，其中重叠率和矛盾率在type为list时，均为0"""
    n = len(data_list)
    # 类覆盖率
    class_type = set([1 if x >= 0.5 else 0 for x in prob])
    class_overlap_rate = len(class_type) / 2.0
    n_matched_samples = 0
    n_overlap_samples = 0
    n_conflict_samples = 0
    for sample in data_list:
        for rule in rule_list:
            if match_onerule_conti(rule, sample, feature_list):
                n_matched_samples += 1
                break
        if type == "set":
            overlap, conflict = match_rules(rule_list, sample, prob, feature_list)
            if overlap > 1:
                n_overlap_samples += 1
            if conflict > 0:
                n_conflict_samples += 1
        # if match_rules(rule_list,sample,prob):
        #   n_overlap_samples += 1
    # print(n_matched_samples,n_overlap_samples,n_conflict_samples)
    coverage_rate = n_matched_samples / n  # 样本覆盖率
    overlap_rate = n_overlap_samples / n  # 重叠率
    conflict_rate = n_conflict_samples / n  # 矛盾率
    return coverage_rate, overlap_rate, conflict_rate, class_overlap_rate


def match_ruleitem(operation, values, flag, feature_value):
    # 匹配一个规则项
    if len(operation) == 1:
        if compare_operation(operation[0], feature_value, float(values[0])) == 0:
            return 0
    if len(operation) == 2:
        if flag:
            temp1 = compare_operation(operation[0], feature_value, float(values[0]))
            temp2 = compare_operation(operation[1], feature_value, float(values[1]))
            if (temp1 or temp2) == 0:
                return 0
        else:  # and
            temp1 = compare_operation(operation[0], float(values[0]), feature_value)
            temp2 = compare_operation(operation[1], feature_value, float(values[1]))
            if (temp1 and temp2) == 0:
                return 0
    return 1

"""稳定性"""
def jaccard_similarity(rule_lists):
    jaccard_list = []
    for i, rule_list in enumerate(rule_lists):
        if i == len(rule_lists) - 1:
            break
        rule_items1 = []
        rule_items2 = []
        for rule in rule_list:
            rule_items1 += re.findall("'([^']*)'", rule)
        for rule in rule_lists[i + 1]:
            rule_items2 += re.findall("'([^']*)'", rule)
        # intersection = len(list(set(rule_list).intersection(rule_lists[i + 1])))
        # union = (len(rule_list) + len(rule_lists[i + 1])) - intersection
        intersection = len(list(set(rule_items1).intersection(rule_items2)))
        union = (len(list(set(rule_items1))) + len(list(set(rule_items2)))) - intersection
        jaccard_list.append(float(intersection) / union)
    if len(jaccard_list) == 0:
        return 0
    else:
        return np.mean(jaccard_list)

def compute_explain_stability(data_list, cover_data_list, rule_list, feature_list):
    df = pd.DataFrame(data_list)
    perturb_range = (df.max() - df.min()).to_dict()  # 扰动范围
    # print(perturb_range)
    cover_data = sum(cover_data_list, [])
    rs = random.sample(cover_data, int(len(cover_data) * 0.05))  # 采样

    new_stability = []
    for sample in rs:
        rule_lists = []
        rule = get_match_rule(rule_list, sample, feature_list)
        rule_lists.append([rule])
        rule_feature = []
        rule_items = re.findall("'([^']*)'", rule)  # 提取规则项
        for item in rule_items:
            feature = feature_name_extract(item, feature_list)  # 提取属性名
            rule_feature.append(feature)
        new_sample = sample.copy()
        for feature in feature_list:
            if feature not in rule_feature:
                new_sample[feature] += random.uniform(perturb_range[feature] * (-0.1), perturb_range[feature] * (0.1))
        rule_lists.append([get_match_rule(rule_list, new_sample, feature_list)])
        new_stability.append(jaccard_similarity(rule_lists))  # 相似性
    stability = sum(new_stability) / len(new_stability)
    return stability


"""因果性"""
def getCausal(x, y_train, rule_list, feature_list):
    df = pd.DataFrame(x, columns=feature_list)
    y = y_train
    ate = []
    for rule in rule_list:
        rule_items = re.findall("'([^']*)'", rule)  # 提取规则项
        if rule_items:
            for item in rule_items:
                feature = feature_name_extract(item, feature_list)  # 提取属性名
                operation = operation_extract(item)  # 提取比较运算符
                values = re.findall("(\d+\.\d*)", item)  # 提取属性值（最后一项为概率值）
                flag = True if ' or ' in item and len(operation) == 2 else False
                treatment = np.array(
                    [(1 if match_ruleitem(operation, values, flag, val) else 0) for val in df[feature].values]).astype(
                    int)
                X = df.drop(feature, axis=1).values
                xg = XGBTRegressor(random_state=42)
                te, lb, ub = xg.estimate_ate(X, treatment, y)

                # nn = MLPTRegressor(hidden_layer_sizes=(10, 10),
                #                   learning_rate_init=.1,
                #                   early_stopping=True,
                #                   random_state=42)
                # te, lb, ub = nn.estimate_ate(X, treatment, y)

                ate.append(abs(te))
    return np.mean(ate)

"""充分性"""
def compute_sufficiency(cover_data_list,rule_list,feature_list,blackbox,weights):
    rule_sufficiency = []
    for i in range(len(cover_data_list)):
        random_size = int(round(len(cover_data_list[i]) * 0.1))  # 样本数的10%，四舍五入取整
        if random_size==0 and len(cover_data_list[i])!=0:
            random_size=1
        rs = random.sample(cover_data_list[i], random_size)
        df = pd.DataFrame()  # 新样本
        df1 = pd.DataFrame()  # 原样本
        rule_feature = []  # 规则中包含的属性名
        rule_items = re.findall("'([^']*)'", rule_list[i])  # 提取规则项
        for item in rule_items:
            feature = feature_name_extract(item, feature_list)  # 提取属性名
            rule_feature.append(feature)
        for sample in rs:
            temp_sample = sample.copy()
            for feature in feature_list:
                if feature not in rule_feature:
                    temp_sample[feature] = 0  # 不相关项赋值为0

            new = pd.DataFrame(temp_sample, index=[i])
            df = df.append(new)
            new = pd.DataFrame(sample, index=[i])
            df1 = df1.append(new)
        # y_rule = blackbox.predict(df)
        y_rule = blackbox.predict(df)  # 根据规则构建的样本预测结果
        y_bb = blackbox.predict(df1)  # 原样本预测结果
        #print(y_rule)
        #print(y_bb)
        rule_sufficiency.append((np.array(y_rule) == np.array(y_bb)).sum() / len(y_bb))
    #print(rule_sufficiency)
    sufficiency = np.sum(np.multiply(np.array(rule_sufficiency), np.array(weights)))
    return sufficiency

"""有效性"""
def compute_effective(new_rulelist,data_list,y,pred_accuracy,feature_list):
    """计算构造的新的规则的有效性，new_rulelist为新的规则列表，data_list为数据集，y为黑盒预测结果
    pred_accuracy为原规则模型预测的准确度
    返回：有效返回1，无效返回0
    """
    new_y = np.array([pred(x, new_rulelist,feature_list) for x in data_list])

    # new_pred_accuracy = roc_auc_score(new_y,np.array(y))
    new_pred_accuracy = accuracy_score(new_y, np.array(y))
    #new_pred_accuracy = (np.array(y) == new_y).sum() / len(y)
    if new_pred_accuracy < pred_accuracy:
        return 1
    return 0

def compute_rule_effectiveness(rule_list, data_list, y_bb, consistency, feature_list):
    count_effective_rule = 0
    for i in range(len(rule_list) - 1):
        new_rulelist = rule_list.copy()
        del new_rulelist[i]
        count_effective_rule += compute_effective(new_rulelist, data_list, y_bb, consistency, feature_list)
        # new_y = np.array([pred(x, new_rulelist, feature_list) for x in data_list])
    rule_effectiveness = count_effective_rule / (len(rule_list) - 1)  # 有效性
    return rule_effectiveness
def compute_rule_item_effectiveness(rule_list,data_list,y,pred_accuracy,feature_list,weights):
    count_effective_rule_item = []
    for i in range(len(rule_list)-1):
        count = 0
        rule_items = re.findall("'([^']*)'", rule_list[i])  # 提取规则项
        if len(rule_items) == 1: #规则长度为1
            new_rulelist=rule_list.copy()
            del new_rulelist[i]
            count +=compute_effective(new_rulelist,data_list,y,pred_accuracy,feature_list)
        elif len(rule_items) >1:
            for rule_item in rule_items:
                new_rulelist = rule_list.copy()
                rule_item = '\'' + rule_item + '\''
                new_rulelist[i]=new_rulelist[i].replace(rule_item,'') #删除规则项
                count += compute_effective(new_rulelist,data_list,y,pred_accuracy,feature_list)
        else:
            print('error')
            break
        count_effective_rule_item.append(count/len(rule_items))
    #print(count_effective_rule_item)
    rule_item_effectiveness=np.sum(np.multiply(np.array(count_effective_rule_item),np.array(weights))) #有效性
    return rule_item_effectiveness


def sort_rulelist(cover_data_list, blackbox,rule_list):
    pre_acc_list = []
    for i in range(len(cover_data_list)):
        df = pd.DataFrame()
        for sample in cover_data_list[i]:
            new = pd.DataFrame(sample, index=[i])
            df = df.append(new)
            y_bb = blackbox.predict(df)
            # y_bb = blackbox.blackbox_pred(df)  # 黑盒预测结果
        y_rule = [1 for _ in range(len(y_bb))]
        acc = (np.array(y_rule) == np.array(y_bb)).sum() / len(y_bb)
        pre_acc_list.append({'acc': acc, 'index': i})
    # print(pre_acc_list)
    pre_acc = sorted(pre_acc_list, key=lambda x: x['acc'], reverse=True)
    # print(pre_acc)
    new_rulelist = [rule_list[x['index']] for x in pre_acc]
    new_rulelist.append(rule_list[-1])
    return new_rulelist


def binary_rule_extraction(proxy_path, data_list, feature_list, y_bb, x_train, y_train, blackbox):
    # 从proxy_path加载规则
    with open(proxy_path, 'r') as f:
        content = f.read()
        rule_list = content.splitlines()  # 规则列表
        rule_type = rule_list[0]  # 规则类型
        rule_list = rule_list[1:]
        prob = []  # 预测概率
        for i in rule_list:
            str_prob = re.findall("\),(\d+\.?\d*)", i)  # 提取概率 list
            if str_prob:
                float_prob = float(str_prob[0])
                prob.append(float_prob)
    """复杂性"""
    model_size = len(rule_list)  # 规则数量
    max_length = 0  # 最大规则长度
    total_rule_length = 0  # 规则总长度
    for i in rule_list:
        rule_items = re.findall("'([^']*)'", i)  # 提取规则项
        if not rule_items:  # 是default规则
            model_size -= 1
        total_rule_length += len(rule_items)
        if len(rule_items) > max_length:
            max_length = len(rule_items)

    """明确性"""
    coverage_rate, overlap_rate, conflict_rate, class_overlap_rate = definiteness(data_list, rule_list, prob,
                                                                                  rule_type, feature_list)

    """一致性"""
    y_rule = np.array([pred(x, rule_list, feature_list) for x in data_list])  # 规则预测结果
    consistency = roc_auc_score(y_rule, y_bb)

    """因果性"""
    ate = getCausal(x_train, y_train, rule_list, feature_list)

    # 统计样本隶属情况
    cover_data_list = [[] for _ in range(len(rule_list) - 1)]  # 样本隶属情况
    for sample in data_list:
        for i in range(len(rule_list) - 1):
            if match_onerule_conti(rule_list[i], sample, feature_list):
                cover_data_list[i].append(sample)
    weights = [len(x) for x in cover_data_list]
    summ = sum(weights)
    weights = [x / summ for x in weights]  # 规则权重

    if rule_type == "set":
        rule_list = sort_rulelist(cover_data_list, blackbox, rule_list)  # 排序后的规则列表



    """稳定性"""
    # 解释稳定性
    explain_stability = compute_explain_stability(data_list, cover_data_list, rule_list, feature_list)

    """充分性"""
    sufficiency = compute_sufficiency(cover_data_list, rule_list, feature_list, blackbox, weights)
    """有效性"""
    # 规则有效性
    rule_effectiveness = compute_rule_effectiveness(rule_list, data_list, y_bb, consistency, feature_list)

    # 规则项有效性
    rule_item_effectiveness = compute_rule_item_effectiveness(rule_list, data_list, y_bb, consistency,
                                                              feature_list, weights)

    index_dict = {"model_size": model_size, "max_length": max_length,
                       "total_rule_length": total_rule_length,
                       "coverage_rate": coverage_rate, "overlap_rate": overlap_rate,
                       "class_overlap_rate": class_overlap_rate,
                       "conflict_rate": conflict_rate, "consistency": consistency,
                       "ATE": ate, "explain_stability": explain_stability,
                       "sufficiency": sufficiency, "rule_effectiveness": rule_effectiveness,
                       "rule_item_effectiveness": rule_item_effectiveness}


    print(index_dict)
    return index_dict


def rule_extract(path_dict, mode):
    """从dataset_test加载样本、特征列表"""
    dataset_test_path = path.join(basedir, path_dict["dataset_test"])
    dataset_train_path = path.join(basedir, path_dict["dataset_train"])
    csv_data = pd.read_csv(dataset_train_path, low_memory=False)  # 防止弹出警告
    x_train = np.array(csv_data)[:, :-1]
    y_train = np.array(csv_data)[:, -1]
    csv_data = pd.read_csv(dataset_test_path, low_memory=False)  # 防止弹出警告
    feature_list = csv_data.columns.values.tolist()[:-1]
    x_test_df = csv_data.iloc[:, :-1]
    x_test = np.array(x_test_df)
    y_test = np.array(csv_data)[:, -1]
    data_list = []  # 组合成数据集列表，列表的每一个项是一个样本，每个项是dict
    for item in x_test:
        data_list_item = dict(zip(feature_list, item))  # 字典类型
        data_list.append(data_list_item)
    """判断分类任务类型"""
    label_size = len(set(list(y_train)))
    """遍历全部XAI提取指标"""
    index_dict = {}
    for i in range(len(path_dict) - 2):
        xai = "XAI" + str(i + 1)
        proxy_path = path.join(basedir, path_dict[xai]["proxy"])
        """ 加载黑盒、规则 """
        if mode == '0':  # 使用黑盒模型
            # 从blackbox_path加载黑盒，获取黑盒预测结果y_bb
            blackbox_path = path.join(basedir, path_dict[xai]["blackbox"])
            blackbox = joblib.load(blackbox_path)
            y_bb = blackbox.predict(x_test)
        else:  # 直接使用数据集
            y_bb = y_test
            if label_size == 2:
                index_dict[xai] = binary_rule_extraction(proxy_path, data_list, feature_list, y_bb, x_train, y_train)
            else:
                index_dict[xai] = multiclass_rule_extraction(proxy_path, x_test_df, feature_list, y_test, x_train,
                                                             y_train)
        return index_dict

"""***************************************************多分类用代码*********************************************"""


def multiclass_feature_name_extract(rule_item):
    # 提取属性名
    # rule_item为一个规则项，str
    # str为要返回的运算符 str
    return re.search("\W*([a-z].*) \W", rule_item).group(1)


def multiclass_match_onerule_conti(rule, sample,feature_list):
    # rule是从文本中提取的一行规则
    # sample是字典列表
    rule_items = re.findall("'([^']*)'", rule)  # 提取规则项
    if rule_items:
        for item in rule_items:
            operation = operation_extract(item)  # 提取比较运算符
            feature = feature_name_extract(item,feature_list)  # 提取属性名
            values = re.findall("(\d+\.\d*)", item)  # 提取属性值（最后一项为概率值）
            # print(operation)
            # print(values)
            n_values = []
            for x in values:
                if '.' in x:
                    n_values.append(x)
            values = n_values  # 筛选掉特征中可能包含的数字
            if len(operation) == 1:
                if compare_operation(operation[0], sample[feature], float(values[0])) == 0:
                    return 0
            if len(operation) == 2:
                if compare_operation(operation[0], float(values[0]), sample[feature]) == 0:
                    return 0
                if compare_operation(operation[1], sample[feature], float(values[1])) == 0:
                    return 0
        return 1
    else:
        return 0


def multiclass_match_rules(rule_list, sample, prob_list,feature_list):
    n = 0
    n_conflict = 0
    class_ = 0
    class_temp = 0
    for i in range(len(rule_list)):
        if multiclass_match_onerule_conti(rule_list[i], sample,feature_list):
            n += 1
            if i <= len(prob_list) - 1:
                if prob_list[i] > 0.5:
                    class_temp = 1
                else:
                    class_temp = 0
            if n > 1:  # 匹配多个规则
                if class_temp != class_:
                    n_conflict += 1
            class_ = class_temp
    return n, n_conflict  # 匹配的规则数，矛盾的次数


def label_cover(prob):
    """类覆盖率，prob：list，是所有规则的预测概率"""
    count = 0
    label_dict = {}
    label_sum = len(prob[0])
    for i in range(label_sum):
        label_dict[i]=False
    for x in prob[:-1]:
        max_index = x.index(max(x))
        if label_dict[max_index] == False:
            label_dict[max_index] = True
            count += 1
            if(count >= label_sum):
                return 1.0
    return count/label_sum


def multiclass_definiteness(data_list, rule_list, prob, type,feature_list):
    """计算明确性指标，其中重叠率和矛盾率在type为list时，均为0"""
    n = len(data_list)
    n_matched_samples = 0
    n_overlap_samples = 0
    n_conflict_samples = 0
    for sample in data_list:  # 计算每一个sample是否能够有匹配规则 计算覆盖率用
        for rule in rule_list:
            if multiclass_match_onerule_conti(rule, sample,feature_list):
                n_matched_samples += 1
                break
        if type == "set":
            overlap, conflict = multiclass_match_rules(rule_list, sample, prob,feature_list)
            if overlap > 1:
                n_overlap_samples += 1
            if conflict > 0:
                n_conflict_samples += 1
    coverage_rate = n_matched_samples / n  # 样本覆盖率
    overlap_rate = n_overlap_samples / n  # 重叠率
    conflict_rate = n_conflict_samples / n  # 矛盾率
    label_coverage_rate = label_cover(prob)  # 类覆盖率
    return coverage_rate, overlap_rate, conflict_rate, label_coverage_rate


def multiclass_pred(x_dict, rule_list, feature_list):
    """预测结果计算"""
    for i, rule in enumerate(rule_list):
        if multiclass_match_onerule_conti(rule, x_dict,feature_list):  # 第一条匹配的规则
            str_prob = re.findall(",(\d+\.?\d*)", rule)  # 提取概率 list
            probs = [float(x) for x in str_prob]
            return probs.index(max(probs))
    str_prob = rule_list[-1].split(',')  # 默认规则
    probs = [float(x) for x in str_prob]
    return probs.index(max(probs))

def multiclass_compute_effective(new_rulelist,data_list,y,pred_accuracy, feature_list):
    """计算构造的新的规则的有效性，new_rulelist为新的规则列表，data_list为数据集，y为黑盒预测结果
    pred_accuracy为原规则模型预测的准确度
    返回：有效返回1，无效返回0
    """
    new_y = np.array([multiclass_pred(x, new_rulelist, feature_list) for x in data_list])
    # AUC_macro = roc_auc_score(new_y, np.array(y), 'macro', multi_class='ovo')
    accuracy = accuracy_score(new_y, y)
    # if AUC_macro < pred_accuracy:
    if accuracy < pred_accuracy:
        return 1
    return 0

def multiclass_compute_rule_effectiveness(rule_list, data_list, y_bb, pred_accuracy, feature_list):
    count_effective_rule = 0
    for i in range(len(rule_list) - 1):
        new_rulelist = rule_list.copy()
        del new_rulelist[i]
        count_effective_rule += multiclass_compute_effective(new_rulelist, data_list, y_bb, pred_accuracy, feature_list)
        # new_y = np.array([pred(x, new_rulelist, feature_list) for x in data_list])
    rule_effectiveness = count_effective_rule / (len(rule_list) - 1)  # 有效性
    return rule_effectiveness

def multiclass_compute_rule_item_effectiveness(rule_list,data_list,y,pred_accuracy,weights, feature_list):
    count_effective_rule_item = []
    for i in range(len(rule_list)-1):
        count = 0
        rule_items = re.findall("'([^']*)'", rule_list[i])  # 提取规则项
        if len(rule_items) == 1: #规则长度为1
            new_rulelist=rule_list.copy()
            del new_rulelist[i]
            count +=multiclass_compute_effective(new_rulelist,data_list,y,pred_accuracy, feature_list)
        elif len(rule_items) >1:
            for rule_item in rule_items:
                new_rulelist = rule_list.copy()
                rule_item = '\'' + rule_item + '\''
                new_rulelist[i]=new_rulelist[i].replace(rule_item,'') #删除规则项
                count += multiclass_compute_effective(new_rulelist,data_list,y,pred_accuracy, feature_list)
        else:
            print('error')
            break
        count_effective_rule_item.append(count/len(rule_items))
    #print(count_effective_rule_item)
    rule_item_effectiveness=np.sum(np.multiply(np.array(count_effective_rule_item),np.array(weights))) #有效性
    return rule_item_effectiveness

def multiclass_rule_extraction(proxy_path, x, feature_list, y_pred_bb, x_train, y_train):
    """读入规则 进行了修改，可以读入多个概率"""
    with open(proxy_path, 'r') as f:
        content = f.read()
        rule_list = content.splitlines()  # 规则列表
        rule_list.pop(0)
        prob = []  # 预测概率list 包括default
        for i, rule in enumerate(rule_list):
            str_prob = re.findall(",(\d+\.?\d*)", rule)  # 提取概率 list 默认rule的也提取了
            if i < len(rule_list) - 1:
                prob.append([float(x) for x in str_prob])
            else:
                rule = rule.split(sep=",")
                prob.append([float(x) for x in rule])
    """复杂性"""
    model_size = len(rule_list)
    max_length = 0  # 最大规则长度
    total_rule_length = 0  # 规则总长度
    for i in rule_list:
        rule_items = re.findall("'([^']*)'", i)  # 提取规则项
        # print(rule_items)
        if not rule_items:  # 是default规则
            model_size -= 1
        total_rule_length += len(rule_items)
        if len(rule_items) > max_length:
            max_length = len(rule_items)

    """覆盖、重叠、矛盾率"""
    X_list = x.values.tolist()  # x.value.tolist()
    data_list = []
    for item in X_list:
        data_list_item = dict(zip(feature_list, item))  # 字典类型
        data_list.append(data_list_item)
    coverage_rate, overlap_rate, conflict_rate, label_coverage_rate = multiclass_definiteness(data_list, rule_list,
                                                                                              prob, "list",feature_list)
    # 统计隶属情况，计算规则权重
    cover_data_list = [[] for _ in range(len(rule_list) - 1)]  # 样本隶属情况
    for sample in data_list:
        for i in range(len(rule_list) - 1):
            if multiclass_match_onerule_conti(rule_list[i], sample, feature_list):
                cover_data_list[i].append(sample)
    weights = [len(x) for x in cover_data_list]
    summ = sum(weights)
    weights = [x / summ for x in weights]  # 规则权重
    print('规则权重', weights)

    """一致性"""
    # 利用规则列表进行预测
    enc = OneHotEncoder()
    y_true = enc.fit_transform(y_pred_bb.reshape(-1, 1)).toarray()
    y = np.array([multiclass_pred(x_dict, rule_list, feature_list) for x_dict in data_list])
    y_pred = enc.transform(y.reshape(-1, 1)).toarray()
    AUC_macro = roc_auc_score(y_true, y_pred, 'macro', multi_class='ovo')
    # AUC_micro = roc_auc_score(y_true, y_pred, 'micro', multi_class='ovo')

    pre_acc = accuracy_score(y_pred, y_true)
    """因果性"""
    ate = getCausal(x_train, y_train, rule_list, feature_list)


    """稳定性"""
    # 解释稳定性
    explain_stability = compute_explain_stability(data_list, cover_data_list, rule_list, feature_list)

    """充分性"""
    #需要用到黑盒模型
    """有效性"""
    rule_effectiveness = multiclass_compute_rule_effectiveness(rule_list, data_list, y_pred_bb, pre_acc, feature_list)
    print('规则有效性：', rule_effectiveness)

    # 规则项有效性
    rule_item_effectiveness = multiclass_compute_rule_item_effectiveness(rule_list, data_list, y_pred_bb, pre_acc,
                                                              weights, feature_list)
    print('规则项有效性：', rule_item_effectiveness)

    index_dict = {"model_size": model_size, "max_length": max_length, "total_rule_length": total_rule_length,
                  "coverage_rate": coverage_rate, "overlap_rate": overlap_rate, "conflict_rate": conflict_rate,
                  "consistency": AUC_macro, "ATE": ate,"sufficiency": 1,
                  "class_overlap_rate": label_coverage_rate,"explain_stability":explain_stability,
                  "rule_effectiveness": rule_effectiveness, "rule_item_effectiveness": rule_item_effectiveness}
    return index_dict


def rule_extract(path_dict, mode):
    """从dataset_test加载样本、特征列表"""
    dataset_test_path = path.join(basedir, path_dict["dataset_test"])
    dataset_train_path = path.join(basedir, path_dict["dataset_train"])
    csv_data = pd.read_csv(dataset_train_path, low_memory=False)  # 防止弹出警告
    x_train = np.array(csv_data)[:, :-1]
    y_train = np.array(csv_data)[:, -1]
    csv_data = pd.read_csv(dataset_test_path, low_memory=False)  # 防止弹出警告
    feature_list = csv_data.columns.values.tolist()[:-1]
    x_test_df = csv_data.iloc[:, :-1]
    x_test = np.array(x_test_df)
    y_test = np.array(csv_data)[:, -1]
    data_list = []  # 组合成数据集列表，列表的每一个项是一个样本，每个项是dict
    for item in x_test:
        data_list_item = dict(zip(feature_list, item))  # 字典类型
        data_list.append(data_list_item)
    """判断分类任务类型"""
    label_size = len(set(list(y_train)))
    """遍历全部XAI提取指标"""
    index_dict = {}
    for i in range(len(path_dict) - 2):
        xai = "XAI" + str(i + 1)
        proxy_path = path.join(basedir, path_dict[xai]["proxy"])
        """ 加载黑盒、规则 """
        if mode == '0':  # 使用黑盒模型
            # 从blackbox_path加载黑盒，获取黑盒预测结果y_bb
            blackbox_path = path.join(basedir, path_dict[xai]["blackbox"])
            blackbox = joblib.load(blackbox_path)
            y_bb = blackbox.predict(x_test)
        else:  # 直接使用数据集
            y_bb = y_test
        if label_size == 2:
            index_dict[xai] = binary_rule_extraction(proxy_path, data_list, feature_list, y_bb, x_train, y_train, blackbox)
        else:
            index_dict[xai] = multiclass_rule_extraction(proxy_path, x_test_df, feature_list, y_test, x_train, y_train)
    return index_dict