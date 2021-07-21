import pandas as pd


def preprocess(df):
    df['cp_type'] = df['cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72:2})
    df = df.drop('sig_id', axis=1)
    return df


def prepare_data(dir):
    X_train = pd.read_csv(f'{dir}/train_features.csv')
    Y_train = pd.read_csv(f'{dir}/train_targets_scored.csv')
    X_test = pd.read_csv(f'{dir}/test_features.csv')
    ss = pd.read_csv(f'{dir}/sample_submission.csv')

    train = X_train.merge(Y_train, on='sig_id')
    X_train = train.loc[:, X_train.columns]
    Y_train = train.loc[:, Y_train.columns]

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    Y_train = Y_train.drop('sig_id', axis=1)

    return X_train, Y_train, X_test, ss