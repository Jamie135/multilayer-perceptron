import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def split(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv('data_train.csv', index=False)
    test.to_csv('data_test.csv', index=False)


def main():
    if len(sys.argv) != 2:
        print('Usage: python3 split.py data.csv')
        sys.exit(1)
    file = sys.argv[1]
    try:
        data = pd.read_csv(file)
        split(data)
    except FileNotFoundError:
        print(f'Error: {file} not found')
        sys.exit(1)


if __name__ == '__main__':
    main()
