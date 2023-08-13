import pandas as pd

from sklearn import feature_selection


def VarianceThreshold(df: pd.DataFrame, threshold=0.0, features=None, inplace=False):
    """
    VarianceThreshold
    """
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str):
        features = [features]
    elif not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")

    other_features = [feature for feature in df.columns.values.tolist() if feature not in features]

    variance = df[features].var()

    selector = feature_selection.VarianceThreshold(threshold=threshold)
    selector.fit(df[features])

    selected = df[features].loc[:, selector.get_support()]
    df = pd.concat([df[other_features], selected], axis=1)

    return {
        "df": df,
        "variance": variance,
        "selected": selected,
        "selected_features": selected.columns.values.tolist(),
        "selected_variance": variance.loc[selector.get_support()],
    }


def UnivariateSelection(df: pd.DataFrame, features=None, target="", k=1,
                        problem_type="classification", scoring="f_classif",
                        inplace=False):
    if problem_type == "classification":
        valid_scoring = {
            "chi2": feature_selection.chi2,
            "f_classif": feature_selection.f_classif,
            "mutual_info_classif": feature_selection.mutual_info_classif,
        }
    elif problem_type == "regression":
        valid_scoring = {
            "f_regression": feature_selection.f_regression,
            "mutual_info_regression": feature_selection.mutual_info_regression,
        }
    else:
        raise Exception("Invalid problem_type. Expected one of: classification, regression")

    if scoring not in valid_scoring:
        raise Exception(f"Invalid scoring. Expected one of: {list(valid_scoring.keys())}")

    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str) or not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")

    if isinstance(k, float):
        k = int(k * len(features))

    k = min(k, len(features))

    other_features = [feature for feature in df.columns.values.tolist() if feature not in features]

    selector = feature_selection.SelectKBest(valid_scoring[scoring], k=k)
    selector.fit(df[features], df[target])

    selected = df[features].loc[:, selector.get_support()]
    df = pd.concat([df[other_features], selected], axis=1)

    scores = pd.DataFrame(selector.scores_, index=features, columns=["score"])
    pvalues = pd.DataFrame(selector.pvalues_, index=features, columns=["pvalue"])

    return {
        "df": df,
        "scores": scores,
        "pvalues": pvalues,
        "selected": selected,
        "selected_features": selected.columns.values.tolist(),
        "selected_scores": scores.loc[selector.get_support()],
        "selected_pvalues": pvalues.loc[selector.get_support()],
    }


def RecursiveFeatureElimination(df: pd.DataFrame, features=None, target="",
                                estimator=None, step=1, inplace=False):
    if not inplace:
        df = df.copy()

    if features is None:
        features = df.columns.values.tolist()
    elif isinstance(features, str) or not hasattr(features, "__iter__"):
        raise Exception("features must be a iteration of features")

    other_features = [feature for feature in df.columns.values.tolist() if feature not in features]

    selector = feature_selection.RFE(estimator=estimator, n_features_to_select=step)
    selector.fit(df[features], df[target])

    selected = df[features].loc[:, selector.get_support()]
    df = pd.concat([df[other_features], selected], axis=1)

    ranking = pd.DataFrame(selector.ranking_, index=features, columns=["ranking"])

    return {
        "df": df,
        "ranking": ranking,
        "selected": selected,
        "selected_features": selected.columns.values.tolist(),
        "selected_ranking": ranking.loc[selector.get_support()],
    }
