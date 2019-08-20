def coverage_score(df, preds, max_runtime=300000):
    solved = 0
    for pred, (index, row) in zip(preds, df.iterrows()):
        if row[pred] < max_runtime:
            solved += 1
    return solved / len(df)


def cumsum_score(df, preds):
    cumsum = 0
    for pred, (index, row) in zip(preds, df.iterrows()):
        cumsum += row[pred]
    return cumsum / (60 * (10 ** 3))
