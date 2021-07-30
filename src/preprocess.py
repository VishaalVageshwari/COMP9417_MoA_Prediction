import pandas as pd


def preprocess(df):
    df['cp_dose'] = df['cp_dose'].map({'D1': 0, 'D2': 1})
    df['cp_time'] = df['cp_time'].map({24: 0, 48: 1, 72:2})
    df = df[df['cp_type']  != 'ctl_vehicle'].reset_index(drop=True)
    return df


def prepare_data(dir):
    X_train = pd.read_csv(f'{dir}/train_features.csv')
    Y_train = pd.read_csv(f'{dir}/train_targets_scored.csv')
    X_test = pd.read_csv(f'{dir}/test_features.csv')
    ss = pd.read_csv(f'{dir}/sample_submission.csv')

    train = X_train.merge(Y_train, on='sig_id')
    Y_train_stub = train.loc[:, Y_train.columns]
    
    train = preprocess(train)
    X_test = preprocess(X_test).drop(['cp_type'], axis=1)

    X_train = train.loc[:, X_train.columns].drop(['sig_id', 'cp_type'], axis=1)
    Y_train = train.loc[:, Y_train.columns]

    return X_train, Y_train, Y_train_stub, X_test, ss