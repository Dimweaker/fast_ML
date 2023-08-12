import pandas as pd
import numpy as np

from itertools import combinations
from fast_ML.Encoder import LabelEncode


def CombinedOneHot(df: pd.DataFrame, features=None, n=2, drop_first=False, inplace=False, brackets="()", sep=", "):
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]

    if n > len(features):
        raise Exception("n cannot be greater than the number of features")

    for comb in combinations(features, n):
        new_feature = brackets[0] + sep.join(comb) + brackets[1]
        df[new_feature] = df[comb].apply(all, axis=1).astype(int)

    if drop_first:
        df.drop(features, axis=1, inplace=True)

    return df


def CombinedLabel(df: pd.DataFrame, features=None, n=2, drop_first=False, inplace=False, mapping=None, brackets="()", sep=", "):
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]
        mapping = {features: mapping}

    if n > len(features):
        raise Exception("n cannot be greater than the number of features")

    combs = combinations(features, n)
    for comb in combs:
        new_feature = brackets[0] + sep.join(comb) + brackets[1]
        df[new_feature] = df[comb].apply(lambda x: "_".join(x), axis=1)

    new_features = [brackets[0] + sep.join(comb) + brackets[1] for comb in combs]
    df = LabelEncode(df, features=new_features, inplace=True, mapping=mapping)

    if drop_first:
        df.drop(features, axis=1, inplace=True)

    return df


def Transform(df: pd.DataFrame, method, features=None, drop_first=False, inplace=False):
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]

    if callable(method):
        for feature in features:
            if method.__name__ == "<lambda>":
                new_feature = f"{feature}_transformed"
            else:
                new_feature = f"{feature}_{method.__name__}"
            df[new_feature] = df[feature].apply(method)
    elif isinstance(method, str):
        for feature in features:
            try:
                new_feature = f"{feature}_{method}"
                df[new_feature] = getattr(np, method)(df[feature])
            except Exception as e:
                raise Exception(e)
    else:
        raise Exception("Invalid method. Expected one of: callable, str")

    if drop_first:
        df.drop(features, axis=1, inplace=True)
