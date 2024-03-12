import argparse
from catboost import CatBoostRanker, Pool
from utils import load_data, prepare_data

def evaluate_model(model_path, data_path):
    model = CatBoostRanker()
    model.load_model(model_path, format='cbm')

    df_test = load_data(data_path)
    test_data = prepare_data(df_test)

    ndcg = model.score(test_data)
    print(f"NDCG value: {ndcg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CatBoostRanker model.")
    parser.add_argument('--model_path', type=str, help='Path to the saved CatBoostRanker model.')
    parser.add_argument('--data_path', type=str, help='Path to the test dataset.')

    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_path)

