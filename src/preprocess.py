import pandas as pd


def preprocess(df):
    df['cp_type'] = df['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72:2})
    df = df.drop('sig_id', axis=1)
    return df


def prepare_data(dir):
    train_features = pd.read_csv(f'{dir}/train_features.csv')
    train_target = pd.read_csv(f'{dir}/train_targets_scored.csv')
    test_features = pd.read_csv(f'{dir}/test_features.csv')

    train_features = preprocess(train_features)
    test_features = preprocess(test_features)

    train_target = train_target.drop('sig_id', axis=1)

    return train_features.values, test_features.values, train_target.values