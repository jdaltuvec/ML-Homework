def prep_data(df):

    df = df.assign(hw=df["Height"] * df["Width"])

    X = df[["Height", "Width", "Length1", "Length2", "Length3", "hw"]].values
    y = df["Weight"].values

    return X, y