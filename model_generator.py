import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import joblib

data = pd.read_csv("dataset/heart.csv")
# print(data.head())

features = ['age', 'sex', 'cp', 'thal', 'trestbps', 'thalach', 'fbs']

X = data[features]
Y = data['target']

# print(X.head())
# print(Y.head())

X_train, x_test, Y_train, y_test = train_test_split(X,Y, random_state=1)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(x_test)
# print(y_pred)

# print("Accuracy of Model : {}" .format(mean_absolute_error(y_test, y_pred)))
print("Accuracy Of Model :  {}" .format(model.score(x_test, y_test)))

joblib.dump(model, 'ml_model.joblib')