import pandas as pd


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

    all_dummies = pd.get_dummies(df[features], drop_first=drop_first, prefix_sep=sep, dummy_na=dummy_na)
    df = pd.concat([df, all_dummies], axis=1)

    return df


def LabelEncode(df: pd.DataFrame, features=None, inplace=False, mapping=None):
    """
    LabelEncoder
    """
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]
        mapping = {features: mapping}

    if mapping is None:
        mapping = {}
        for feature in features:
            mapping[feature] = df[feature].unique().tolist()
            mapping[feature].sort()
            mapping[feature] = {k: v for v, k in enumerate(mapping[feature])}

    for feature in features:
        df[feature] = df[feature].map(mapping[feature])

    return df