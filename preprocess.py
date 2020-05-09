def prep_data(df):

    df = df.assign(hw = df["Height"] * df["Width"] )
    df = df.assign(all_l = (df["Length1"] * df["Length2"] * df["Length3"])/3 )

    X = df[["hw", "all_l"]].values
    y = df["Weight"].values

    return X, y