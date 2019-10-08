# -*- coding: utf-8 -*-
# Name:

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn import svm


global x_simple
global y_label


x_train, x_test, y_train, y_test = model_selection.train_test_split(x_simple, y_label, test_size=0.2, random_state=0)

log_model = LogisticRegression()
log_model.fit(x_train, y_train)

log_model.score(x_test, y_test)

y_pred = log_model.predict(x_test)




# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovo')
clf = svm.SVC(C=0.9, kernel='rbf', gamma=20, decision_function_shape='ovo')
clf.fit(x_train, y_train)

clf.score(x_train, y_train) # 精度
clf.score(x_test, y_test)

y_pred = clf.predict(x_test)


# 查看测试结果
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

