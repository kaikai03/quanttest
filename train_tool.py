# -*- coding: utf-8 -*-
# Name:

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB

from sklearn.model_selection import GridSearchCV

global x_simple
global y_label


x_train, x_test, y_train, y_test = model_selection.train_test_split(x_simple, y_label, test_size=0.8, random_state=0)

x_train = x_simple[0:int(len(x_simple)*0.9)]
x_test = x_simple[int(len(x_simple)*0.9):]
y_train = y_label[0:int(len(y_label)*0.9)]
y_test = y_label[int(len(y_label)*0.9):]


log_model = LogisticRegression()
log_model.fit(x_train, y_train)

log_model.score(x_test, y_test)

y_pred = log_model.predict(x_test)

##########################################


# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovo')
clf = svm.SVC(C=1.0, kernel='rbf', gamma=150, decision_function_shape='ovo')
clf.fit(x_train, y_train)

clf.score(x_train, y_train) # 精度
clf.score(x_test, y_test)

y_pred = clf.predict(x2_simple[-22:])

print(metrics.confusion_matrix(y2_label[-22:], y_pred))
print(metrics.classification_report(y2_label, y_pred))
metrics.mean_squared_error(y2_label, y_pred)
metrics.accuracy_score(y2_label[-22:], y_pred)
metrics.roc_auc_score(y2_label, clf.predict_proba(x2_simple)[:,1])

clf.feature_importances_
############################################

# clf = RandomForestRegressor
# clf = ExtraTreesClassifier()
# clf = DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=300)
clf.fit(x_train,y_train)


clf.score(x_train, y_train) # 精度
clf.score(x_test, y_test)

y_pred=clf.predict(x2_simple)


############################################


clf = MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(120, 100, 2), alpha=0.0001, batch_size='auto',max_iter=500, verbose=1)
clf.fit(x_train,y_train)
clf.score(x_test, y_test)



clf.score(x_train, y_train) # 精度
clf.score(x_test, y_test)

clf.loss_

y_pred=clf.predict(x_test)

###############################################

clf = MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(120,100,100,100,50,8,2,), alpha=0.0001, batch_size='auto',max_iter=500)
param_grid = {'hidden_layer_sizes': [(20,20),(50,50), (100,100), (100,100,), (120,100,2),(120,100,100,), (120,100,100,50,),
                                     (120,100,100,50,8), (120,100,100,50,8,2), (120,100,100,50,8,2,)]}
grid_search = GridSearchCV(clf, param_grid, n_jobs = 3, verbose=10)
grid_search.fit(x_simple, y_label)
grid_search.best_estimator_.get_params()['hidden_layer_sizes']


############################################


# clf = GradientBoostingClassifier(n_estimators=500)
clf = AdaBoostClassifier(n_estimators=200,algorithm='SAMME')
clf.fit(x_train,y_train)


clf.score(x_train, y_train) # 精度
clf.score(x_test, y_test)

y_pred=clf.predict(x2_simple)


############################################

clf = BernoulliNB()
clf = MultinomialNB()
clf = GaussianNB()
clf.fit(x_train,y_train)

clf.score(x_train, y_train) # 精度
clf.score(x_test, y_test)

y_pred=clf.predict(x_test)


############################################


# 查看测试结果
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
metrics.mean_squared_error(y_test, y_pred)



