import pandas as pd
from joblib import dump
from preprocess import prep_data

df = pd.read_csv("fish_participant.csv")

X, y = prep_data(df)

### Linear regression ###
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X,y)

### Naive Bayes ###
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(X, y)

### Random forest ###
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X,y)
# Keeping random forest—it has the best score on the prediction

dump(model, "reg.joblib")