import sys
import pandas as pd


def splitList(data, cut):
    """
    Splits an array in two
    """
    return data[:cut], data[cut:]


def split(data):
    """
    Split data into train and test sets
    """

    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # Calculate the index to split at 80%
    cut = int(len(data) * 0.8)
    train, test = splitList(data, cut)
    train.to_csv('data/data_training.csv', index=False)
    test.to_csv('data/data_test.csv', index=False)


def main():
    file = "data/data.csv"
    try:
        data = pd.read_csv(file)
        split(data)
    except FileNotFoundError:
        print(f'Error: {file} not found')
        sys.exit(1)


if __name__ == '__main__':
    main()
