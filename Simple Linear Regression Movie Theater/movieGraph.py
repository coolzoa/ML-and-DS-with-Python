#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:13:05 2020

@author: josemurillo
"""

import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

data = pandas.read_csv('cost-revenue-clean.csv')
#data.describe()

X = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])

regression = LinearRegression()
regression.fit(X, Y)
#slope coefficient, Theta1 regression.coef_ 
#y intercept coefficient, Theta 0 regression.intercept_
#gives R2 test value regression.score(X, Y)

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.3)
plt.plot(X, regression.predict(X), color='red', linewidth=4)
plt.title('Film cost vs. Global revenue')
plt.xlabel('Production budget $')
plt.ylabel('Worldwide gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)


plt.show()

