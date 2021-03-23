import pandas as pd
from sklearn.metrics import accuracy_score


def runtime_adjusted_coverage_score(df, preds, max_runtime_arr=None):
    if max_runtime_arr is None:
        max_runtime_arr = [300000] * len(preds)
    tmp_df = df.copy()
    tmp_df['CurrP'] = preds
    if len(set(preds)) == 1:  # Same one, can just use x[preds[0]]
        tmp_df['CurrP-Runtime'] = tmp_df[preds[0]]
    else:
        tmp_df['CurrP-Runtime'] = tmp_df.apply(lambda x: x[x['CurrP']], axis=1)
    solved = (tmp_df['CurrP-Runtime'] <= max_runtime_arr).sum()
    return solved / len(tmp_df)


def coverage_score(df, preds, max_runtime=300000):
    tmp_df = df.copy()
    tmp_df['CurrP'] = preds
    if len(set(preds)) == 1:  # Same one, can just use x[preds[0]]
        if isinstance(preds, pd.Series):
            tmp_df['CurrP-Runtime'] = tmp_df[preds.iloc[0]]
        else:
            tmp_df['CurrP-Runtime'] = tmp_df[preds[0]]
    else:
        tmp_df['CurrP-Runtime'] = tmp_df.apply(lambda x: x[x['CurrP']], axis=1)
    solved = len(tmp_df[tmp_df['CurrP-Runtime'] < max_runtime])
    return solved / len(tmp_df)


def normalized_coverage_score(df, preds, max_runtime=300000):
    tmp_df = df.copy()
    tmp_df['CurrP'] = preds
    if len(set(preds)) == 1:  # Same one, can just use x[preds[0]]
        if isinstance(preds, pd.Series):
            tmp_df['CurrP-Runtime'] = tmp_df[preds.iloc[0]]
        else:
            tmp_df['CurrP-Runtime'] = tmp_df[preds[0]]
    else:
        tmp_df['CurrP-Runtime'] = tmp_df.apply(lambda x: x[x['CurrP']], axis=1)
    normalized_coverage_per_map = tmp_df.groupby('GridName')['CurrP-Runtime'].apply(
        lambda x: (x < 300000).sum() / len(x))
    return normalized_coverage_per_map.mean()


def normalized_accuracy_score(df, preds):
    tmp_df = df.copy()
    if not isinstance(preds, list):
        preds = preds.copy().to_list()
    return tmp_df.reset_index().groupby('GridName').apply(
        lambda x: accuracy_score(x['Y'], [preds[i] for i in list(x.index)])).mean()


def cumsum_score(df, preds, online_feature_extraction_time=None):
    tmp_df = df.copy()
    tmp_df['CurrP'] = preds
    if len(set(preds)) == 1:  # Same one, can just use x[preds[0]]
        if isinstance(preds, pd.Series):
            tmp_df['CurrP-Runtime'] = tmp_df[preds.iloc[0]]
        else:
            tmp_df['CurrP-Runtime'] = tmp_df[preds[0]]
    else:
        tmp_df['CurrP-Runtime'] = tmp_df.apply(lambda x: x[x['CurrP']], axis=1)

    if online_feature_extraction_time:
        # Add online feature calculation time to the cumsum
        cumsum = tmp_df['CurrP-Runtime'].sum() + tmp_df[online_feature_extraction_time].sum()
    else:
        cumsum = tmp_df['CurrP-Runtime'].sum()
    return cumsum // (60 * (10 ** 3))
