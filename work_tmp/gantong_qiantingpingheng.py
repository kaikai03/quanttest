import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import BaggingClassifier




age_train = [8, 5, 9, 10, 6, 9, 10, 5, 7, 8, 7, 8, 8, 9, 5, 7, 8, 6, 6, 7, 7, 7, 10, 8, 5, 8, 9, 7, 10, 8, 6, 7, 9, 9, 9, 9, 7, 3, 10, 9, 6, 5, 6, 9, 10, 7, 9, 9, 7, 10, 9, 7, 9, 8, 9, 10, 7, 7, 9, 7, 7, 8, 8, 10, 9, 10, 10, 10, 8, 6, 6, 6, 5, 5, 7, 8, 5, 6, 6, 12, 9, 7, 9, 9, 12, 5, 8, 7, 7, 5, 6, 9, 7, 6, 8, 7, 10, 8, 10, 7, 9, 6, 8, 6, 8, 8, 5, 6, 5, 5, 5, 7, 12, 6, 7, 7, 5, 7, 8, 7, 7, 5, 10, 5, 7, 5, 5, 11, 10, 11, 9, 8, 11, 8, 6, 4, 8]

score_train = [47, 48, 64, 58, 62, 32, 58, 46, 48, 65, 58, 41, 65, 51, 45, 38, 60, 46, 28, 59, 52, 53, 42, 61, 53, 62, 62, 70, 50, 63, 40, 43, 57, 57, 61, 60, 39, 51, 61, 51, 49, 60, 59, 55, 50, 52, 57, 66, 58, 61, 66, 45, 50, 62, 40, 57, 41, 52, 58, 51, 47, 66, 53, 57, 63, 60, 59, 31, 58, 57, 26, 48, 50, 50, 53, 68, 52, 63, 63, 55, 51, 66, 62, 66, 61, 48, 54, 49, 50, 46, 40, 47, 50, 65, 55, 54, 56, 56, 56, 56, 63, 56, 55, 53, 53, 63, 59, 45, 44, 30, 58, 50, 67, 48, 65, 50, 40, 56, 47, 64, 68, 43, 64, 49, 39, 59, 46, 64, 59, 61, 62, 47, 58, 66, 57, 53, 53]

y_train = [2, 2, 3, 3, 3, 0, 3, 2, 2, 3, 3, 0, 3, 2, 2, 0, 3, 2, 0, 3, 2, 3, 0, 3, 3, 3, 3, 3, 2, 3, 0, 0, 3, 3, 3, 3, 0, 3, 3, 2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 0, 2, 3, 0, 3, 0, 2, 3, 2, 0, 3, 3, 3, 3, 3, 3, 0, 3, 3, 0, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 3, 2, 2, 2, 0, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 3, 2, 3, 2, 3, 2, 0, 3, 2, 3, 3, 0, 3, 2, 0, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3]
y_train = [30, 36, 56, 46, 57, 10, 46, 33, 31, 60, 49, 21, 60, 36, 31, 13, 52, 33, 10, 51, 38, 40, 20, 53, 43, 55, 53, 69, 33, 57, 24, 22, 45, 45, 51, 50, 15, 40, 51, 36, 37, 54, 52, 42, 33, 38, 45, 59, 49, 51, 59, 26, 34, 55, 19, 44, 19, 38, 47, 36, 29, 62, 40, 44, 55, 49, 47, 10, 49, 49, 10, 36, 39, 39, 40, 65, 42, 58, 58, 39, 36, 63, 53, 59, 49, 36, 42, 33, 35, 33, 24, 30, 35, 61, 44, 42, 43, 45, 43, 45, 55, 48, 44, 43, 40, 57, 52, 31, 30, 10, 51, 35, 59, 36, 61, 35, 24, 45, 30, 60, 67, 28, 56, 37, 15, 52, 33, 54, 47, 49, 53, 30, 46, 62, 49, 43, 40]

age = [12, 10, 10, 10, 11, 7, 8, 7, 5, 8, 6, 6, 7, 5, 5, 6, 5, 10, 12, 9, 10, 12, 9, 10, 7, 9, 10, 7, 7, 7, 6, 7, 10]

score = [60, 54, 37, 56, 68, 66, 45, 47, 53, 61, 59, 50, 53, 54, 52, 49, 44, 54, 63, 45, 59, 48, 55, 55, 60, 70, 64, 57, 58, 51, 61, 57, 46]

y = [3, 2, 0, 3, 3, 3, 0, 0, 3, 3, 3, 2, 3, 3, 3, 2, 2, 2, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 0]
y= [47, 39, 12, 43, 61, 63, 27, 29, 43, 53, 52, 39, 40, 45, 42, 37, 30, 39, 52, 26, 47, 27, 42, 41, 53, 65, 56, 47, 49, 36, 55, 47, 26]


x_train = [[x, score_train[i]] for i, x in enumerate(age_train)]


poly = PolynomialFeatures(degree = 3)
x_train_poly = poly.fit_transform(x_train)

x_test = [[x,score[i]] for i,x in enumerate(age)]
x_test_poly = poly.fit_transform(x_test)


line = LinearRegression()
line.fit(x_train, y_train)
line.score(x_test, y)
line.predict(x_test)

line.fit(x_train_poly, y_train)
line.score(x_test_poly, y)
line.predict(x_test_poly)

# plt.plot(y_sample, line.predict(x_poly), color = 'red')
# plt.title('Linear Regression')
# plt.xlabel('Temperature')
# plt.ylabel('Pressure')
# plt.show()

np.sum((np.round(line.predict(x_train_poly)) == np.array(y_train))) / len(y_train)


np.sum((np.round(line.predict(x_test_poly)) == np.array(y)))/len(y)


svm = SVC(kernel='rbf',probability=True)
svm.fit(x_train, y_train)
svm.score(x_test, y)
svm.predict(x_train)
svm.predict_proba([[5,48]])

svm.fit(x_train_poly, y_train)
svm.score(x_test_poly, y)


ridge = Ridge(alpha=0.1).fit(x_train_poly, y_train)
np.sum((np.round(ridge.predict(x_test_poly)) == np.array(y)))/len(y)

GBDT = GradientBoostingClassifier(n_estimators=25)
GBDT.fit(x_train, y_train)
GBDT.score(x_test, y)
GBDT.predict(x_test)

GBDT.fit(x_train_poly, y_train)
GBDT.score(x_test_poly, y)

tr = tree.DecisionTreeClassifier()
tr.fit(x_train, y_train)
tr.score(x_test, y)


forest = RandomForestClassifier(n_estimators=150)
forest.fit(x_train, y_train)
forest.score(x_test, y)
forest.predict([[9,64]])
forest.predict(x_test)
forest.predict(x_train)


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
knn.score(x_test, y)

bag = BaggingClassifier(tree.DecisionTreeClassifier(),n_estimators=5)
bag.fit(x_train, y_train)
bag.score(x_test, y)

lsvm = LinearSVC()
lsvm.fit(x_train, y_train)
lsvm.score(x_train, y_train)
lsvm.predict([[9,32]])

import m2cgen as m2c
code = m2c.export_to_java(line)


# 目前随机森林可正确输出


std_score = [31.05903901, 35.6467055, 56.21375853, 45.96094005, 56.1830505, 6.951096957, 45.96094005, 32.56778915, 33.61456662, 58.76928614, 49.00914836, 21.82228996, 58.76928614, 36.20080227, 31.02833098, 18.21998488, 51.07199527, 31.55171971, 3.841472577, 50.54860654, 39.77239932, 41.31185749, 21.32960926, 52.61145345, 43.34399637, 54.15091162, 53.13484218, 67.48264645, 33.64527465, 55.69036979, 22.31497067, 25.91727575, 45.43755131, 45.43755131, 51.59538401, 50.05592583, 19.75944305, 42.2972189, 50.57931457, 36.20080227, 36.17009423, 54.12020359, 51.56467598, 42.35863496, 33.64527465, 39.77239932, 45.43755131, 59.29267488, 49.00914836, 50.57931457, 59.29267488, 28.9961921, 34.66134409, 54.15091162, 19.26676235, 44.42148187, 22.8383594, 39.77239932, 46.97700948, 38.23294114, 32.07510845, 60.30874432, 40.29578805, 44.42148187, 54.67430036, 49.03985639, 47.50039822, 4.395569344, 47.99307892, 48.48575963, 0.762556229, 34.63063606, 38.72562185, 38.72562185, 41.31185749, 63.38766066, 41.8045382, 57.72250867, 57.72250867, 39.31042665, 36.20080227, 61.32481376, 53.13484218, 59.29267488, 48.54717569, 35.6467055, 41.83524623, 35.1540248, 36.69348297, 32.56778915, 22.31497067, 30.04296957, 36.69348297, 60.80142502, 43.3747044, 42.85131567, 42.8820237, 44.91416258, 42.8820237, 45.93023201, 54.67430036, 46.94630145, 43.3747044, 42.32792693, 40.29578805, 55.69036979, 52.58074541, 30.01226154, 29.4888728, 7.936458365, 51.04128724, 36.69348297, 57.78392473, 34.63063606, 59.78535558, 36.69348297, 23.33104011, 45.93023201, 31.05903901, 58.24589741, 64.4037301, 27.94941463, 55.19768909, 37.18616367, 19.75944305, 52.58074541, 32.56778915, 54.18161965, 47.50039822, 49.56324513, 53.13484218, 31.05903901, 44.94487061, 60.30874432, 48.48575963, 44.36006581, 40.29578805]
std_y = [1, 1, 2, 2, 2, 0, 2, 1, 1, 2, 2, 0, 2, 1, 1, 0, 2, 1, 0, 2, 1, 2, 0, 2, 2, 2, 2, 2, 1, 2, 0, 0, 2, 2, 2, 2, 0,
         2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 0, 2, 0, 1, 2, 1, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 1, 1, 1,
         2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 2,
         1, 2, 1, 2, 1, 0, 2, 1, 2, 2, 0, 2, 1, 0, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2]

std_score = [[i] for i in std_score]

std_test = [46.87082508, 39.22323768, 13.26209561, 42.5384208, 60.89375507, 62.34818491, 26.76377171, 31.02736741, 43.12942386, 53.06951937, 52.03838954, 37.02584372, 40.90473172, 44.80515812, 41.45635237, 35.3764117, 28.41329698, 39.22323768, 51.75496425, 25.68929995, 47.5002417, 27.19731879, 42.00098624, 40.88069154, 52.56009223, 66.35103837, 55.66874471, 47.57799369, 49.24283451, 37.57988219, 55.3515307, 47.57799369, 26.1835551]
std_test = [[i] for i in std_test]

forest = RandomForestClassifier(n_estimators=3)
forest.fit(std_score, std_y)
forest.score(std_score, std_y)
forest.predict(std_score)
forest.predict(std_test)












