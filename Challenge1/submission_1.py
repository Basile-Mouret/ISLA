import argparse
import os
from model_1 import Model

import pandas as pd


def resolve_input_path(input_dir, split, filename):
    direct_path = os.path.join(input_dir, filename)
    nested_path = os.path.join(input_dir, split, filename)
    if os.path.exists(direct_path):
        return direct_path
    if os.path.exists(nested_path):
        return nested_path
    raise FileNotFoundError(f"Could not find {filename} under {input_dir}")


def get_train_data(input_dir):
    X_train = pd.read_csv(resolve_input_path(input_dir, 'train', 'X_train.csv'))
    y_train = pd.read_csv(resolve_input_path(input_dir, 'train', 'y_train.csv'))
    return X_train, y_train


def get_test_data(input_dir):
    X_test = pd.read_csv(resolve_input_path(input_dir, 'test', 'X_test.csv'))
    return X_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print('Reading Data')
    X_train, y_train = get_train_data(input_dir)
    X_test = get_test_data(input_dir)
    print('Starting')
    m = Model()
    print('Training Model')
    m.fit(X_train, y_train)
    print('Running Prediction')
    prediction = m.predict(X_test)
    df = pd.DataFrame(prediction, columns=['age'])
    output_path = os.path.join(output_dir, 'y_pred.csv')
    df.to_csv(output_path, index=False)
    print(f'Wrote predictions to {output_path}')


if __name__ == '__main__':
    main()
