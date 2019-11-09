def coverage_score(df, preds, max_runtime=300000):
    tmp_df = df.copy()
    tmp_df['CurrP'] = preds
    tmp_df['CurrP-Runtime'] = tmp_df.apply(lambda x: x[x['CurrP']], axis=1)
    solved = len(tmp_df[tmp_df['CurrP-Runtime'] < max_runtime])
    return solved/len(tmp_df)


def cumsum_score(df, preds):
    tmp_df = df.copy()
    tmp_df['CurrP'] = preds
    tmp_df['CurrP-Runtime'] = tmp_df.apply(lambda x: x[x['CurrP']], axis=1)
    cumsum = tmp_df['CurrP-Runtime'].sum()
#     cumsum = 0
#     for pred, (index, row) in zip(preds,df.iterrows()):
#         cumsum += row[pred]
    return cumsum/((60*(10**3)))

