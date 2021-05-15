import argparse
import pandas as pd
from utils import get_config


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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)
