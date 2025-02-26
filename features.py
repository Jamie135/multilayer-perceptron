import os
import sys
import csv
import math
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
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
