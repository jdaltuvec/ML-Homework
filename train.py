import pandas as pd
from sklearn.naive_bayes import GaussianNB
from joblib import dump
from preprocess import prep_data

df = pd.read_csv("fish_participant.csv")

X, y = prep_data(df)

gnb = GaussianNB()
gnb.fit(X, y)

dump(gnb, "reg.joblib")