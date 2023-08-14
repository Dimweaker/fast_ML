import pandas as pd
import numpy as np

from itertools import combinations
from sklearn import preprocessing
from fast_ML.Encoder import LabelEncode


def CombinedOneHot(df: pd.DataFrame, features=None, n=2,
                   drop_first=False, inplace=False,
                   brackets="()", sep=", "):
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str) or not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")
    elif len(features) < 2:
        raise Exception("features must contain at least 2 features")

    if n > len(features):
        raise Exception("n cannot be greater than the number of features")

    for comb in combinations(features, n):
        new_feature = brackets[0] + sep.join(comb) + brackets[1]
        df[new_feature] = df[list(comb)].apply(all, axis=1).astype(int)

    if drop_first:
        df.drop(features, axis=1, inplace=True)

    return df


def CombinedLabel(df: pd.DataFrame, features=None, n=2,
                  drop_first=False, inplace=False, mapping=None,
                  brackets="()", sep=", "):
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str) or not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")
    elif len(features) < 2:
        raise Exception("features must contain at least 2 features")

    if n > len(features):
        raise Exception("n cannot be greater than the number of features")

    combs = combinations(features, n)
    new_features = []
    for comb in combs:
        new_feature = brackets[0] + sep.join(comb) + brackets[1]
        new_features.append(new_feature)
        df[new_feature] = df[list(comb)].astype(str).apply(lambda x: "_".join(x), axis=1)

    df = LabelEncode(df, features=new_features, mapping=mapping, inplace=True)

    if drop_first:
        df.drop(features, axis=1, inplace=True)

    return df


def Transform(df: pd.DataFrame, method, features=None,
              drop_first=False, inplace=False,
              on_original_cols=False, prefix="", suffix=""):
    if not inplace:
        df = df.copy()

    if drop_first and not on_original_cols:
        raise Exception("drop_first cannot be True if on_original_cols is False")

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]
    elif not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")

    if callable(method):
        for feature in features:
            if on_original_cols:
                new_feature = feature
            elif any([prefix, suffix]):
                new_feature = f"{prefix}{feature}{suffix}"
            elif method.__name__ == "<lambda>":
                new_feature = f"{feature}_transformed"
            elif hasattr(method, "__name__"):
                new_feature = f"{feature}_{method.__name__}"
            else:
                raise Exception("Can't find a default name. Please provide a name for the method.")

            df[new_feature] = df[feature].apply(method)
    elif isinstance(method, str):
        for feature in features:
            try:
                new_feature = f"{feature}_{method}" if not on_original_cols else feature
                df[new_feature] = getattr(np, method)(df[feature])
            except Exception as e:
                raise Exception(e)
    else:
        raise Exception("Invalid method. Expected one of: callable, str")

    if drop_first:
        df.drop(features, axis=1, inplace=True)

    return df


def Polynomial(df: pd.DataFrame, features=None, degree=2,
               interaction_only=False, include_bias=False, inplace=False):
    if not inplace:
        df = df.copy()

    if not isinstance(degree, int) or degree < 2:
        raise Exception("degree must be an integer greater than 1")

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str) or not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")
    elif len(features) < 2:
        raise Exception("features must contain at least 2 features")

    pf = preprocessing.PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    new_features_data = pf.fit_transform(df[features])[..., len(features):]
    new_features = pf.get_feature_names_out(features)[len(features):]
    df[new_features] = pd.DataFrame(new_features_data, columns=new_features)

    return df


def Bin(df: pd.DataFrame, features=None, bins=10, labels=None,
        right=True, retbins=False, precision=3, include_lowest=False,
        inplace=False, sep_feature_bin="_", sep_bin_n=""):
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]
    elif not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")

    if labels is None:
        labels = range(bins)

    for feature in features:
        new_feature = f"{feature}{sep_feature_bin}bin{sep_bin_n}{bins}"
        df[new_feature], bins = pd.cut(df[feature], bins=bins, labels=labels, right=right,
                                       retbins=retbins, precision=precision, include_lowest=include_lowest)
        if retbins:
            df[new_feature] = df[new_feature].cat.codes

    return df

