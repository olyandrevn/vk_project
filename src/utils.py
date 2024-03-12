import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from catboost import Pool

def load_data(filepath):
    """Load data from a file."""
    return pd.read_csv(filepath)

def split_data(df, group_col='search_id', train_size=0.8, random_state=42):
    """Split data into training and validation sets, preserving groups."""
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    for train_idx, val_idx in gss.split(df, groups=df[group_col]):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
    return df_train, df_val

def prepare_data(df, label_col='target', group_id_col='search_id'):
    """Prepare data, leaving only important features and creating Catboost.Pool object."""
    features = [f'feature_{i}' for i in range(79)]
    leave_features = features[5:25] + features[51:73] + features[77:78]
    return Pool(
        data=df[leave_features],
        label=df[label_col],
        group_id=df[group_id_col]
    )

