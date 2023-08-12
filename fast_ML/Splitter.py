import pandas as pd

from sklearn import model_selection


def CreateFolds(df: pd.DataFrame, target, k=5, shuffle=False, method="kfold", inplace=False):
    """
    Create folds
    """
    if not inplace:
        df = df.copy()

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    if method == "kfold":
        kf = model_selection.KFold(n_splits=k)
    elif method == "stratified":
        kf = model_selection.StratifiedKFold(n_splits=k)
    elif method == "group":
        kf = model_selection.GroupKFold(n_splits=k)
    else:
        raise Exception("Invalid method. Expected one of: kfold, stratified, group")

    y = df[target].values

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    return df
