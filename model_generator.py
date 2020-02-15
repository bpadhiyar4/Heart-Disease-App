import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import joblib

data = pd.read_csv("dataset/heart.csv")

FEATURES = ['age', 'sex', 'cp', 'thal', 'trestbps', 'thalach', 'fbs']

X = data[FEATURES]
Y = data['target']

X_train, x_test, Y_train, y_test = train_test_split(X,Y, random_state=1)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(x_test)

accuracy = model.score(x_test, y_test)
print("Accuracy Of Model :  {}" .format(model.score(x_test, y_test)))

joblib.dump(model, 'ml_model.joblib')