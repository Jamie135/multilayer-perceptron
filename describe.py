import os
import sys
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def load_data():
    with open("./data/data.csv") as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    feature_list = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concavePoints",
        "symmetry",
        "fractalDimension",
    ]
    feat_type = ["Mean", "Standard error", "Largest"]

    return input_data_list, feature_list, feat_type


def calculate_std(input_data_list, col_num, mean, count):
    """
    Calculate the standard deviation of the dataset
    """

    diff_sum = 0
    for data in input_data_list:
        if data[col_num] != "":
            value = float(data[col_num])
            diff_sum += (value - mean) ** 2

    return math.sqrt(diff_sum / (count - 1))


def calculate_threshold_by_ratio(input_data_list, col_num, count, ratio):
    """
    Calculate the threshold of the dataset
    """

    value_list = []
    for data in input_data_list:
        if data[col_num] != "":
            value = float(data[col_num])
            value_list.append(value)

    threshold_index = count * ratio
    value_list.sort()

    return value_list[int(threshold_index)]


def display_result(
    numerical_feature_list,
    count_list,
    mean_list,
    std_list,
    min_list,
    max_list,
    threshold_25_list,
    threshold_50_list,
    threshold_75_list,
):
    """
    Display the statistics of the dataset
    """

    count_list.insert(0, "Count")
    mean_list.insert(0, "Mean")
    std_list.insert(0, "Std")
    min_list.insert(0, "Min")
    threshold_25_list.insert(0, "25%")
    threshold_50_list.insert(0, "50%")
    threshold_75_list.insert(0, "75%")
    max_list.insert(0, "Max")
    data = [
        count_list,
        mean_list,
        std_list,
        min_list,
        threshold_25_list,
        threshold_50_list,
        threshold_75_list,
        max_list,
    ]

    print(tabulate(data, headers=numerical_feature_list))


def describe():
    """
    Display the statistics of the dataset
    """

    input_data_list, feature_list, feat_type = load_data()

    for idx in range(len(feat_type)):
        feature_start_pos = 2
        count_list = []
        mean_list = []
        std_list = []
        min_list = []
        max_list = []
        threshold_25_list = []
        threshold_50_list = []
        threshold_75_list = []
        for i in range(len(feature_list) * len(feat_type)):
            if i % len(feat_type) != idx:
                continue
            col_num = i + feature_start_pos
            count = 0
            data_sum = 0.0
            min_value = sys.float_info.max
            max_value = sys.float_info.min
            for data in input_data_list:
                if data[col_num] != "":
                    value = float(data[col_num])
                    if value < min_value:
                        min_value = value
                    if value > max_value:
                        max_value = value
                    data_sum += value
                    count += 1

            mean = data_sum / count
            std = calculate_std(input_data_list, col_num, mean, count)
            threshold_25 = calculate_threshold_by_ratio(
                input_data_list, col_num, count, 0.25
            )
            threshold_50 = calculate_threshold_by_ratio(
                input_data_list, col_num, count, 0.5
            )
            threshold_75 = calculate_threshold_by_ratio(
                input_data_list, col_num, count, 0.75
            )

            count_list.append(count)
            mean_list.append(mean)
            std_list.append(std)
            min_list.append(min_value)
            max_list.append(max_value)
            threshold_25_list.append(threshold_25)
            threshold_50_list.append(threshold_50)
            threshold_75_list.append(threshold_75)

        print("FEATURE TYPE: ", feat_type[idx])
        display_result(
            feature_list,
            count_list,
            mean_list,
            std_list,
            min_list,
            max_list,
            threshold_25_list,
            threshold_50_list,
            threshold_75_list,
        )
        print()


def scatter():
    """
    Create scatterplot of the dataset
    """

    input_data_list, feature_list, feat_type = load_data()

    fig = plt.figure(figsize=(16, 9))

    malignant = "M"
    benign = "B"
    feature_start_pos = 2
    feat_type_idx = 0
    for i in range(len(feature_list) * len(feat_type)):
        if i % len(feat_type) != feat_type_idx:
            continue
        feat_idx = int(i / len(feat_type))
        col_num = i + feature_start_pos
        malignant_list = []
        benign_list = []
        for data in input_data_list:
            value = float(data[col_num])
            if data[1] == malignant:
                malignant_list.append(value)
            elif data[1] == benign:
                benign_list.append(value)
            else:
                raise RuntimeError("Wrong label")

        graph = fig.add_subplot(2, 5, feat_idx + 1)
        graph.scatter(range(len(malignant_list)), malignant_list, alpha=0.4, label="malignant", color="red")
        graph.scatter(range(len(benign_list)), benign_list, alpha=0.4, label="benign", color="blue")
        graph.legend(loc="upper right", fontsize="7")
        graph.set_title(feature_list[feat_idx], fontsize=10)

    plt.show()


def histogram():
    """
    Create histogram of the dataset
    """

    input_data_list, feature_list, feat_type = load_data()

    fig = plt.figure(figsize=(16, 9))

    malignant = "M"
    benign = "B"
    feature_start_pos = 2
    feat_type_idx = 0
    for i in range(len(feature_list) * len(feat_type)):
        if i % len(feat_type) != feat_type_idx:
            continue
        feat_idx = int(i / len(feat_type))
        col_num = i + feature_start_pos
        malignant_list = []
        benign_list = []
        for data in input_data_list:
            value = float(data[col_num])
            if data[1] == malignant:
                malignant_list.append(value)
            elif data[1] == benign:
                benign_list.append(value)
            else:
                raise RuntimeError("Wrong label")

        graph = fig.add_subplot(2, 5, feat_idx + 1)
        graph.hist(malignant_list, alpha=0.4, label="malignant", color="red")
        graph.hist(benign_list, alpha=0.4, label="benign", color="blue")
        graph.legend(loc="upper right", fontsize="7")
        graph.set_title(feature_list[feat_idx], fontsize=10)

    plt.show()


def main():
    try:
        describe()
        scatter()
        histogram()
    except FileNotFoundError as f:
        print(f)
        sys.exit(1)
    except ValueError as v:
        print(v)
        sys.exit(1)


if __name__ == "__main__":
    main()
