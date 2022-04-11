import math
import operator

import TreePlotter


def create_data():
    """

    :return: 训练数据集 特征标签
    """
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    return datasets, labels


def split_dataset(datasets, index, value):
    """

    :param datasets:
    :param index: 特征对应索引
    :param value: 特征取值
    :return: 按特征的取值划分的数据集
    """
    res_dataset = []
    for fea in datasets:
        if fea[index] == value:
            res_feat = fea[:index]
            res_feat.extend(fea[index:])
            res_dataset.append(res_feat)
    return res_dataset


def cal_shannon_entropy(datasets):
    """

    :param datasets:
    :return: 数据集的经验熵
    """
    num_entries = len(datasets)
    label_counter = {}
    for feat in datasets:
        if feat[-1] not in label_counter.keys():
            label_counter[feat[-1]] = 0
        label_counter[feat[-1]] += 1

    shannon_entropy = 0.0
    for label in label_counter.keys():
        prob = float(label_counter[label]) / num_entries
        shannon_entropy -= prob * math.log(prob, 2)
    return shannon_entropy


def choose_best_feature_to_split(datasets):
    """

    :param datasets:
    :return: 最佳划分的特征对应的索引
    """
    num_feat = len(datasets[0]) - 1
    base_entropy = cal_shannon_entropy(datasets)
    best_info_gain = 0
    best_feat = -1

    for feat_idx in range(num_feat):
        feat_list = [info[feat_idx] for info in datasets]
        unique_vals = set(feat_list)
        condition_entropy = 0
        for val in unique_vals:
            sub_dataset = split_dataset(datasets, feat_idx, val)
            prob = len(sub_dataset) / float(len(datasets))
            condition_entropy += prob * cal_shannon_entropy(sub_dataset)
        info_gain = base_entropy - condition_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = feat_idx
    return best_feat


def creat_tree(datasets, labels):
    """
    :param datasets: 数据集
    :param labels: 特征标签
    :return: 构建的决策树
    """
    class_list = [info[-1] for info in datasets]

    # 所有样本取值一致
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 所有特征都用过了
    if len(datasets[0]) == 1:
        counter = {}
        for value in class_list:
            if value not in counter.keys():
                counter[value] = 0
            counter[value] += 1
        sorted_class_count = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0]

    best_feat = choose_best_feature_to_split(datasets)
    best_feat_label = labels[best_feat]
    tree = {best_feat_label: {}}
    del(labels[best_feat])

    feat_values = [info[best_feat] for info in datasets]
    unique_feat_values = set(feat_values)

    for value in unique_feat_values:
        sub_labels = labels[:]
        tree[best_feat_label][value] = creat_tree(
            split_dataset(datasets, best_feat, value), sub_labels)

    return tree


if __name__ == "__main__":
    datasets, labels = create_data()
    tree = creat_tree(datasets, labels)

    print(tree)
    TreePlotter.createPlot(tree)
