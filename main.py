import numpy as np
import matplotlib.pyplot as plt

# using loadtxt()
df = np.loadtxt("/content/FuelConsumptionCo2.csv",
                 delimiter=",", dtype=str)
display(df)



enginesize = df[1:, 4].astype('float64')
print(enginesize)

fc_comb = df[1:, 10].astype('float64')
print(fc_comb)

co2 = df[1:, 12].astype('float64')
print(co2)



class LinearRegression:
    weights = []
    intercept = 0

    def __init__(self):
        self.weights = []
        self.intercept = 0

    def fit(self, X, y):
        y = y.astype('float64')
        ms = []
        for x in X:
            x = x.astype('float64')
            a = sum((x - np.mean(x)) * (y - np.mean(y))) / sum((x - np.mean(x)) ** 2)
            self.weights.append(a)
            ms.append(a * np.mean(x))
        b = np.mean(y) - sum(ms)
        self.intercept = b

    def predict(self, X):
        ms = []
        for i in range(len(self.weights)):
            x = X[i].astype('float64')
            ms.append(self.weights[i] * np.mean(x))
        y = sum(ms) + self.intercept
        print(y)


l1 = LinearRegression()
l1.fit(np.array([enginesize, fc_comb]), co2)
l1.predict(np.array([enginesize, fc_comb]))
