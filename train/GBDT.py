#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import os
import numpy as np
from sklearn import linear_model, cross_validation, ensemble, grid_search
import codecs


parent_path = os.path.dirname(os.getcwd())
data_path = os.path.join(parent_path, 'original_data')
file_in_X = csv.reader(open(os.path.join(data_path, 'train_x.csv'), 'rb'))     # Load in the csv file
header_X = file_in_X.next()                             # Skip the fist line as it is a header
X = []
for row in file_in_X:
    temp = []
    for i in range(1, len(row)) :
        temp.append(float(row[i]))
    X.append(temp)
X = np.array(X)
file_in_y = csv.reader(open(os.path.join(data_path, 'train_y.csv'), 'rb'))
header_y = file_in_y.next()
y = []
for row in file_in_y:
    temp = []
    for i in range(1, len(row)) :
        temp.append(float(row[i]))
    y.append(temp)
y = np.array(y)
y = np.ravel(y)

parameters = {'max_depth': range(5, 50, 5), 'n_estimators':[100, 200, 400, 800, 1000, 1500, 2000],
              'max_features': ["sqrt", "log2", "auto"], 'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001]}
GBRT = ensemble.GradientBoostingRegressor()
clf = grid_search.GridSearchCV(GBRT, parameters, cv = 10)


clf.fit(X, y)

file_in_test = csv.reader(open(os.path.join(data_path, 'test_x.csv'), 'rb'))     # Load in the csv file
header_test = file_in_test.next()                             # Skip the fist line as it is a header
test_X = []
uids = []
for row in file_in_test:
    temp = []
    uids.append(row[0])
    for i in range(1, len(row)) :
        temp.append(float(row[i]))
    test_X.append(temp)
test_X = np.array(test_X)

out = clf.predict(test_X)


file_predict = codecs.open('../predict/predict.csv', 'w', 'utf-8')
csv_writer = csv.writer(file_predict)
csv_writer.writerow(['"uid"', '"score"'])
for i in xrange(len(uids)):
    csv_writer.writerow([uids[i], float(out[i])])
file_predict.close()


