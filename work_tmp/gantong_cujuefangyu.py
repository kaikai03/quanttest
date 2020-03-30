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


age_train = [8, 5, 9, 10, 6, 9, 10, 5, 7, 8, 7, 8, 8, 9, 5, 7, 8, 6, 6, 7, 7, 7, 10, 8, 5, 8, 9, 7, 10, 8, 6, 7, 9, 9, 9, 9, 7, 3, 10, 9, 6, 5, 6, 9, 10, 7, 9, 9, 7, 10, 9, 7, 9, 8, 9, 10, 7, 7, 9, 7, 7, 8, 8, 10, 9, 10, 10, 10, 8, 6, 6, 6, 5, 5, 7, 8, 5, 6, 6, 12, 9, 7, 9, 9, 12, 5, 8, 7, 7, 5, 6, 9, 7, 6, 8, 7, 10, 8, 10, 7, 9, 6, 8, 6, 8, 8, 5, 6, 5, 5, 5, 7, 12, 6, 7, 7, 5, 7, 8, 7, 7, 5, 10, 5, 7, 5, 5, 11, 10, 11, 9, 8, 11, 8, 6, 4, 8]
score_train = [99, 66, 105, 94, 105, 70, 102, 90, 103, 99, 104, 56, 101, 81, 69, 72, 93, 72, 49, 99, 100, 86, 90, 96, 85, 103, 98, 105, 78, 87, 69, 72, 98, 93, 93, 93, 80, 83, 92, 68, 80, 98, 103, 86, 97, 82, 93, 105, 103, 103, 102, 81, 75, 95, 75, 81, 84, 83, 102, 86, 84, 95, 80, 83, 95, 87, 101, 94, 105, 79, 52, 79, 93, 70, 98, 102, 96, 81, 99, 96, 83, 101, 96, 97, 104, 93, 97, 70, 96, 86, 71, 86, 91, 84, 100, 73, 96, 88, 83, 98, 95, 96, 99, 91, 98, 95, 90, 75, 78, 75, 85, 80, 101, 91, 99, 79, 77, 73, 82, 103, 100, 83, 95, 87, 79, 84, 70, 102, 90, 88, 81, 91, 105, 94, 87, 75, 80]
x_train = [[x, score_train[i]] for i, x in enumerate(age_train)]
x_train = [[x, score_train[i], x**2, x*score_train[i], score_train[i]**2,
            x**6.2+score_train[i]**0.55,score_train[i]**8] for i, x in enumerate(age_train)]
x_train = [[x, score_train[i], x**2, x*score_train[i], score_train[i]**2] for i, x in enumerate(age_train)]


final_y_train = [2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 1, 2, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 0, 1, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
y_train = [59, 26, 65, 52, 65, 29, 61, 50, 65, 59, 66, 10, 62, 40, 29, 30, 52, 32, 10, 60, 61, 46, 48, 56, 45, 64, 57, 67, 36, 45, 29, 30, 57, 52, 52, 52, 39, 43, 50, 27, 40, 58, 63, 45, 56, 41, 52, 65, 65, 62, 62, 40, 34, 55, 34, 39, 44, 43, 62, 46, 44, 55, 37, 41, 54, 45, 60, 52, 67, 39, 12, 39, 53, 30, 59, 63, 56, 41, 59, 55, 42, 63, 55, 56, 64, 53, 57, 28, 57, 46, 31, 45, 51, 44, 61, 32, 55, 46, 41, 59, 54, 56, 59, 51, 58, 55, 50, 35, 38, 35, 45, 39, 61, 51, 60, 38, 37, 32, 39, 65, 61, 43, 54, 47, 38, 44, 30, 62, 48, 46, 40, 50, 64, 53, 47, 35, 37]

age_test = [12, 10, 10, 10, 11, 7, 8, 7, 5, 8, 6, 6, 7, 5, 5, 6, 5, 10, 12, 9, 10, 12, 9, 10, 7, 9, 10, 7, 7, 7, 6, 7, 10]
score_test = [101, 90, 79, 86, 100, 96, 80, 92, 90, 58, 94, 83, 89, 79, 68, 82, 72, 76, 93, 99, 88, 86, 81, 94, 94, 91, 105, 96, 98, 82, 105, 102, 84]
x_test = [[x,score_test[i]] for i,x in enumerate(age_test)]
x_test = [[x, score_test[i], x**2, x*score_test[i], score_test[i]**2,
           x**6.2+score_test[i]**0.55,score_test[i]**8] for i, x in enumerate(age_test)]
x_test = [[x, score_test[i], x**2, x*score_test[i], score_test[i]**2] for i, x in enumerate(age_test)]

y_test = [61, 48, 37, 44, 59, 57, 37, 53, 50, 11, 54, 43, 47, 39, 28, 42, 32, 34, 51, 58, 46, 43, 40, 52, 55, 50, 64, 57, 59, 41, 65, 64, 42]


final_y = [2, 2, 1, 2, 2, 2, 1, 2, 2, 0, 2, 2, 2, 1, 0, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]




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













