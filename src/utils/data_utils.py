import os
import json
import joblib
import argparse
import sys
import pandas as pd
import numpy as np
from utils import get_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet


def get_data_from_source(config_path):
    data_path = get_config(config_path, 'data_source', 's3_source')
    data = pd.read_csv(data_path)
    return data


def load_and_save(config_path):
    data = get_data_from_source(config_path)
    columns = data.columns
    new_cols = [col.replace(" ", "_") for col in columns]
    raw_data_path = get_config(config_path, 'load_data', 'raw_dataset_csv')
    data.to_csv(raw_data_path, sep=",", index=False, header=new_cols)


def split_and_save_data(config_path):
    test_data_path = get_config(config_path, 'split_data', 'test_path')
    train_data_path = get_config(config_path, 'split_data', 'train_path')
    raw_data_path = get_config(config_path, 'load_data', 'raw_dataset_csv')
    split_ratio = get_config(config_path, 'split_data', 'test_size')
    random_state = get_config(config_path, 'base', 'random_state')
    data = pd.read_csv(raw_data_path, sep=',')
    test, train = train_test_split(data, test_size=split_ratio, random_state=random_state)
    test.to_csv(test_data_path, sep=",", index=False, encoding='utf-8')
    train.to_csv(train_data_path, sep=",", index=False, encoding='utf-8')


def eva_matrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def save_scores(config_path, rmse, mae, r2):
    scores_file = get_config(config_path, "reports", "scores")
    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)


def save_params(config_path, alpha, l1_ratio):
    params_file = get_config(config_path, "reports", "params")

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        }
        json.dump(params, f, indent=4)


def train_and_evaluate(config_path):
    test_data_path = get_config(config_path, 'split_data', 'test_path')
    train_data_path = get_config(config_path, 'split_data', 'train_path')
    random_state = get_config(config_path, 'base', 'random_state')
    model_dir = get_config(config_path, 'model_dir')
    alpha = get_config(config_path, 'estimators', 'ElasticNet', 'params', 'alpha')
    l1_ratio = get_config(config_path, 'estimators', 'ElasticNet', 'params', 'l1_ratio')
    target = [get_config(config_path, 'base', 'target_col')]
    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')
    train_y = train[target]
    test_y = test[target]
    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    en.fit(train_x, train_y)
    predicted_qualities = en.predict(test_x)
    (rmse, mae, r2) = eva_matrics(test_y, predicted_qualities)
    save_scores(config_path, rmse, mae, r2)
    save_params(config_path, alpha, l1_ratio)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(en, model_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--stage", default="load_and_save")
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    stage = parsed_args.stage
    getattr(sys.modules[__name__], stage)(config_path=parsed_args.config)
