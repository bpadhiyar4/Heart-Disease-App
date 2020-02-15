import pandas as pd
import numpy as np

data = pd.read_csv("dataset/heart.csv")
# print(data.head())

feature = ['age', 'sex', 'thal']
X = data[feature]
Y = data['target']

print(X)
print(Y)