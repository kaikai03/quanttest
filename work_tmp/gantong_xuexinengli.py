import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# import matplotlib.pyplot as plt
# import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import BaggingClassifier


age_train = [8, 9, 10, 6, 9, 10, 7, 8, 7, 8, 8, 9, 7, 8, 6, 6, 7, 7, 7, 10, 8, 8, 9, 7, 10, 8, 6, 7, 9, 9, 9, 9, 7, 10, 9, 6, 6, 9, 10, 7, 9, 9, 7, 10, 9, 7, 9, 8, 9, 10, 7, 7, 9, 7, 7, 8, 8, 10, 9, 10, 10, 10, 8, 6, 6, 6, 7, 8, 6, 6, 12, 9, 7, 9, 9, 12, 8, 7, 7, 6, 9, 7, 6, 8, 7, 10, 8, 10, 7, 9, 6, 8, 6, 8, 8, 6, 7, 12, 6, 7, 7, 7, 8, 7, 7, 10, 7, 11, 10, 11, 9, 8, 11, 8, 6]
score_train = [35, 31, 27, 33, 16, 27, 16, 28, 18, 25, 37, 33, 24, 24, 24, 24, 20, 24, 25, 14, 38, 27, 31, 21, 19, 32, 29, 20, 35, 29, 18, 24, 9, 36, 23, 27, 31, 19, 22, 20, 24, 16, 34, 40, 34, 23, 22, 26, 21, 20, 24, 25, 37, 29, 35, 26, 21, 27, 19, 30, 23, 25, 20, 24, 9, 23, 25, 38, 32, 38, 37, 27, 31, 27, 37, 28, 24, 23, 18, 19, 27, 21, 30, 19, 22, 20, 26, 27, 30, 25, 25, 27, 25, 16, 16, 26, 27, 35, 18, 32, 27, 26, 27, 38, 34, 17, 28, 26, 22, 26, 26, 36, 36, 27, 26]
x_train = [[x, score_train[i]] for i, x in enumerate(age_train)]
x_train = [[x, score_train[i], x**2, x*score_train[i], score_train[i]**2,
            x**6.2+score_train[i]**0.55,score_train[i]**8] for i, x in enumerate(age_train)]
x_train = [[x, score_train[i], x**2, x*score_train[i], score_train[i]**2] for i, x in enumerate(age_train)]


final_y_train = [2, 2, 1, 2, 0, 1, 0, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 2, 0, 0, 2, 2, 0, 2, 2, 0, 1, 0, 2, 1, 1, 2, 0, 1, 0, 1, 0, 2, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 2, 2, 1, 0, 1, 0, 2, 1, 1, 0, 1, 0, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 1, 1, 2, 0, 2, 1, 1, 1, 2, 2, 0, 2, 1, 1, 1, 1, 2, 2, 1, 1]
y_train = [54, 46, 39, 49, 19, 39, 17, 41, 21, 35, 58, 50, 33, 33, 32, 32, 25, 33, 35, 16, 60, 39, 46, 27, 25, 49, 41, 25, 53, 42, 23, 33, 10, 55, 32, 37, 45, 25, 30, 25, 33, 19, 52, 62, 51, 31, 30, 37, 28, 26, 33, 35, 57, 42, 54, 37, 28, 39, 25, 44, 32, 35, 26, 32, 10, 30, 35, 60, 47, 58, 56, 39, 46, 39, 57, 40, 33, 31, 21, 22, 39, 27, 43, 24, 29, 26, 37, 39, 44, 35, 34, 39, 34, 18, 18, 35, 38, 52, 20, 48, 38, 36, 39, 60, 52, 21, 40, 37, 30, 37, 37, 56, 55, 39, 35]

age_test = [8, 12, 10, 10, 10, 11, 7, 8, 7, 8, 6, 6, 7, 6, 10, 12, 9, 10, 12, 9, 10, 7, 9, 10, 7, 7, 7, 6, 7, 10]
score_test = [21, 30, 27, 17, 20, 32, 36, 33, 20, 10, 31, 29, 24, 25, 18, 25, 22, 28, 34, 24, 26, 13, 29, 38, 25, 31, 23, 38, 29, 17]
x_test = [[x,score_test[i]] for i,x in enumerate(age_test)]
x_test = [[x, score_test[i], x**2, x*score_test[i], score_test[i]**2,
           x**6.2+score_test[i]**0.55,score_test[i]**8] for i, x in enumerate(age_test)]
x_test = [[x, score_test[i], x**2, x*score_test[i], score_test[i]**2] for i, x in enumerate(age_test)]

y_test = [28, 44, 39, 21, 26, 47, 56, 50, 25, 10, 45, 41, 33, 34, 23, 35, 30, 41, 51, 33, 37, 11, 42, 59, 35, 46, 31, 58, 42, 21]
final_y = [0, 2, 1, 0, 0, 2, 2, 2, 0, 0, 2, 2, 1, 1, 0, 1, 1, 2, 2, 1, 1, 0, 2, 2, 1, 2, 1, 2, 2, 0]




poly = PolynomialFeatures(degree=2 ,include_bias=False)
poly.fit(x_train, y_train)
x_train_poly = poly.fit_transform(x_train)

x_test_poly = poly.fit_transform(x_test)


line = LinearRegression()
line.fit(x_train, y_train)
line.score(x_test, y_test)
line.predict(x_test)
line.predict(x_train)


line.fit(x_train_poly, y_train)
line.score(x_test_poly, y)
line.predict(x_test_poly)

# plt.plot(y_sample, line.predict(x_poly), color = 'red')
# plt.title('Linear Regression')
# plt.xlabel('Temperature')
# plt.ylabel('Pressure')
# plt.show()





forest = RandomForestClassifier(n_estimators=150)
forest.fit(x_train, y_train)
forest.score(x_test, y)
forest.predict([[9,64]])
forest.predict(x_test)
forest.predict(x_train)



import m2cgen as m2c
code = m2c.export_to_java(line)
code = m2c.export_to_java(forest)


# 目前随机森林可正确输出


std_score = line.predict(x_train)
std_score = [[i] for i in std_score]

std_test = line.predict(x_test)
std_test = [[i] for i in std_test]

forest = RandomForestClassifier(n_estimators=3)
forest.fit(std_score, final_y_train)
forest.score(std_score, final_y_train)

forest.score(std_test, final_y)



for i in range(1,99):
    forest = RandomForestClassifier(n_estimators=i)
    forest.fit(std_score, final_y_train)
    print(i, forest.score(std_test, final_y))













