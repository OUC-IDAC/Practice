# ID3 Decision Tree
import math
import json
from sklearn.datasets import load_iris


def create_example_data_set():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def create_iris_data_set():
    iris = load_iris()
    data_set = iris.data.tolist()
    target_list = iris.target
    target_names = iris.target_names.tolist()
    index = 0
    target_names_dict = dict()
    for name in target_names:
        target_names_dict.setdefault(index, name)
        index = index + 1
    for i in range(len(data_set)):
        data_set[i].append(target_names_dict[target_list[i]])
    for fields in data_set:
        for i in range(len(fields)):
            fields[i] = str(fields[i])
    labels = iris.feature_names
    return data_set, labels


def plot_iris_3d():
    import plotly.plotly as py
    import plotly.graph_objs as go
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    iris_feature_names = iris.feature_names
    trace = go.Scatter3d(
        x=iris_data[:, 2],
        y=iris_data[:, 1],
        z=iris_data[:, 0],
        mode='markers',
        marker=dict(
            size=6,
            color=iris_target,
            colorscale='Viridis',
            opacity=0.8
        )
    )
    layout = go.Layout(
        title='Iris',
        scene=dict(
            xaxis=dict(
                autorange=True,
                showline=False,
                backgroundcolor='rgb(105, 105, 105)',
                showbackground=True,
                gridwidth=2,
                gridcolor="rgb(255, 255, 255)",
                zerolinewidth=4,
                zerolinecolor="rgb(255, 255, 255)",
                linecolor="rgb(211, 211, 211)",
                linewidth=5,
                title=iris_feature_names[2]
            ),
            yaxis=dict(
                autorange=True,
                showline=False,
                backgroundcolor='rgb(105, 105, 105)',
                showbackground=True,
                gridwidth=2,
                gridcolor="rgb(255, 255, 255)",
                zerolinewidth=4,
                zerolinecolor="rgb(255, 255, 255)",
                linecolor="rgb(211, 211, 211)",
                linewidth=5,
                title=iris_feature_names[1]
            ),
            zaxis=dict(
                autorange=True,
                showline=False,
                backgroundcolor='rgb(105, 105, 105)',
                showbackground=True,
                gridwidth=2,
                gridcolor="rgb(255, 255, 255)",
                zerolinewidth=4,
                zerolinecolor="rgb(255, 255, 255)",
                linecolor="rgb(211, 211, 211)",
                linewidth=5,
                title=iris_feature_names[0]
            )
        )
    )
    fig = go.Figure(data=[trace], layout=layout)
    py.iplot(fig)


def create_watermelon_data_set():
    data_set = list()
    with open('dataset/watermelon_dataset.csv', 'r', encoding='utf-8') as f:
        for field in f.readlines():
            field_list = field.split(',')
            field_list[-1] = field_list[-1][:-1]
            data_set.append(field_list)
    with open('dataset/watermelon_labels.csv', 'r', encoding='utf-8') as f:
        labels = f.readline().split(',')

    return data_set, labels


def calc_shannon_entropy(data_set):
    num = len(data_set)
    label_dict = dict()
    for data in data_set:
        label = data[-1]
        if label not in label_dict:
            label_dict.setdefault(label, 0)
        label_dict[label] += 1
    entropy = 0.0
    for key in label_dict:
        probability = float(label_dict[key]) / num
        entropy += probability * math.log2(1 / probability)
    return entropy


def split_data_set(data_set, axis, value):
    split_set = list()
    for data in data_set:
        if data[axis] == value:
            reduced_set = data[:axis]
            reduced_set.extend(data[axis + 1:])
            split_set.append(reduced_set)
    return split_set


def choose_best_split_axis(data_set):
    axis_num = len(data_set[0]) - 1
    print('Axis num: ', axis_num)
    base_entropy = calc_shannon_entropy(data_set)
    best_information_gain = 0.0
    best_split_axis = -1

    for axis in range(axis_num):
        value_list = [data[axis] for data in data_set]
        value_set = set(value_list)
        new_entropy = 0.0

        for value in value_set:
            sub_data_set = split_data_set(data_set, axis, value)
            probability = len(sub_data_set) / float(len(data_set))
            new_entropy += probability * calc_shannon_entropy(sub_data_set)

        information_gain = base_entropy - new_entropy
        print(information_gain)

        if information_gain > best_information_gain:
            best_information_gain = information_gain
            best_split_axis = axis

    print('Best split axis: ', best_split_axis)
    return best_split_axis


def get_most_label(label_list):
    label_dict = dict()
    for label in label_list:
        if label not in label_dict.keys():
            label_dict.setdefault(label, 0)
        label_dict[label] += 1
    sorted_label_dict = sorted(label_dict.items(), key=lambda i: i[1], reverse=True)
    # print(sorted_label_dict)
    return sorted_label_dict[0][0]


def create_tree(data_set, labels):
    label_list = [data[-1] for data in data_set]
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]

    if len(data_set[0]) == 1:
        return get_most_label(label_list)

    best_split_axis = choose_best_split_axis(data_set)
    best_axis_label = labels[best_split_axis]
    print(labels)

    tree = {best_axis_label: {}}
    # print(best_axis_label)
    del(labels[best_split_axis])

    axis_value_set = set([data[best_split_axis] for data in data_set])
    for value in axis_value_set:
        sub_labels = labels[:]
        try:
            tree[best_axis_label][value] = create_tree(split_data_set(data_set, best_split_axis, value), sub_labels)
        except IndexError:
            print(best_split_axis)
    return tree


def classify(tree, feature_labels, feature):
    first_feature = list(tree.keys())[0]
    sub_dict = tree[first_feature]

    feature_index = feature_labels.index(first_feature)
    class_label = None
    for key in sub_dict:
        if feature[feature_index] == key:
            if type(sub_dict[key]).__name__ == 'dict':
                class_label = classify(sub_dict[key], feature_labels, feature)
            else:
                class_label = sub_dict[key]
    return class_label


def accuracy(predicted, targets):
    num = 0
    for i in range(len(predicted)):
        if predicted[i] == targets[i]:
            num = num + 1
    return float(num) / len(predicted)


def main():
    data_set, labels = create_watermelon_data_set()
    # plot_iris_3d()

    # print(data_set)
    # print(labels)
    tree = create_tree(data_set, labels)
    print(tree)
    print(json.dumps(tree, indent=3, sort_keys=True,
                      ensure_ascii=False))
    data_set, labels = create_iris_data_set()
    predicted = [classify(tree, labels, data_set[i][:-1]) for i in range(len(data_set))]
    correct = [data_set[i][-1] for i in range(len(data_set))]

    print('Train Accuracy: {0}'.format(accuracy(predicted, correct)))


main()