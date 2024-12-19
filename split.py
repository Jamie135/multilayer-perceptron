import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    """split a dataset into training and test sets"""
    if len(sys.argv) != 2:
        print('Usage: python3 split.py data.csv')
        sys.exit(1)

    file = sys.argv[1]
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f'Error: {file} not found')
        sys.exit(1)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv('data_training.csv', index=False)
    test.to_csv('data_testing.csv', index=False)


if __name__ == '__main__':
    main()
