import pandas as pd
import argparse
from catboost import CatBoostRanker
from utils import load_data, split_data, prepare_data

def fit_model(train_data, val_data, loss_function, additional_params=None):
    parameters = {
        'iterations': 1000,
        'verbose': False,
        'allow_writing_files': False,
        'random_seed': 0,
        'loss_function': loss_function,
    }
    if additional_params:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_data, eval_set=val_data)
    return model


def main(data_path, model_path, loss_function='YetiRank'):
    df = load_data(data_path)
    df_train, df_val = split_data(df)
    train_data = prepare_data(df_train)
    val_data = prepare_data(df_val)

    model = fit_model(train_data, val_data, loss_function)
    model.save_model(model_path, format='cbm')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CatBoostRanker model.')
    parser.add_argument('--data_path', type=str, help='Path to the train dataset.')
    parser.add_argument('--model_path', type=str, help='Path to save the model.')
    parser.add_argument('--loss_function', type=str, default='YetiRank', help='Loss function to be used for training.')

    args = parser.parse_args()

    main(args.data_path, args.model_path, args.loss_function)
