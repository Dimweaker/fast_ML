import pandas as pd

from sklearn import preprocessing


def OneHotEncode(df: pd.DataFrame, features=None, drop_first=False, inplace=False, sep="_", dummy_na=False):
    """
    OneHotEncoder
    """
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]
    elif not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")

    all_dummies = pd.get_dummies(df[features], drop_first=drop_first, prefix_sep=sep, dummy_na=dummy_na)
    df = pd.concat([df, all_dummies], axis=1)

    return df


def LabelEncode(df: pd.DataFrame, features=None,
                inplace=False, mapping=None,
                on_original_cols=True, prefix="", suffix="_labeled", ):
    """
    LabelEncoder
    """
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        if mapping is not None:
            mapping = {features: mapping}
        features = [features]
    elif not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")

    # if mapping is None:
    #     mapping = {}
    #     for feature in features:
    #         mapping[feature] = df[feature].unique().tolist()
    #         mapping[feature].sort()
    #         mapping[feature] = {k: v for v, k in enumerate(mapping[feature])}

    new_features = [f"{prefix}{feature}{suffix}" if not on_original_cols else feature
                    for feature in features]

    if mapping is None:
        le = preprocessing.LabelEncoder()
        df[new_features] = df[features].apply(le.fit_transform)
    else:
        df[new_features] = df[features].apply(lambda x: x.map(mapping[x.name]))

    return df
