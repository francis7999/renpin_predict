#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn import cross_validation, ensemble
from sklearn.feature_selection import SelectFromModel
from sklearn.semi_supervised import LabelSpreading
import codecs
from sklearn.externals import joblib
import time
import csv

data_path = '../original_data'
Labeled_X = pd.read_csv(os.path.join(data_path, 'train_x.csv'))
Labeled_y = pd.read_csv(os.path.join(data_path, 'train_y.csv'))
Unlabeled_X_1 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_1.csv'))
Unlabeled_X_2 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_2.csv'))
#Unlabeled_X = pd.concat([Unlabeled_X_1, Unlabeled_X_2],join='inner',ignore_index=True)
Unlabeled_y_1 = -1 * np.ones((len(Unlabeled_X_1)))
Unlabeled_y_2 = -1 * np.ones((len(Unlabeled_X_2)))
del Unlabeled_X_1['uid']
del Unlabeled_X_2['uid']
del Labeled_X['uid']
del Labeled_y['uid']
a_Labeled_y = np.array(Labeled_y['y'])

clf = joblib.load('GBRT.pkl')
print('\n导入成功\n')
model = SelectFromModel(clf, prefit = True, threshold = "1.5*mean")
Labeled_X_new = model.transform(Labeled_X)
Unlabeled_X_1_new = model.transform(Unlabeled_X_1)
Unlabeled_X_2_new = model.transform(Unlabeled_X_2)
print Unlabeled_X_1_new.shape
print Unlabeled_X_2_new.shape
Unlabeled_X_new = np.concatenate((Unlabeled_X_1_new, Unlabeled_X_2_new), axis = 0)

del Labeled_X
del Unlabeled_X_1
del Unlabeled_X_2
del Unlabeled_X_1_new
del Unlabeled_X_2_new
train_X = np.concatenate((Labeled_X_new, Unlabeled_X_new), axis = 0)
train_y = np.concatenate((a_Labeled_y, Unlabeled_y_1, Unlabeled_y_2), axis = 0)
label_spread = LabelSpreading('knn')
t1 = time.time()
label_spread.fit(train_X, train_y)
t2 = time.time()
print t2-t1
t3 = time.time()
GBRT2 = ensemble.GradientBoostingRegressor(n_estimators= 200, max_depth = 50, max_features= 'sqrt')
GBRT2.fit(train_X, label_spread.transduction_)
t4 = time.time()
test_X = pd.read_csv(os.path.join(data_path, 'test_x.csv'))
uids = test_X['uid']
del test_X['uid']
out= GBRT2.predict(test_X)

file_predict = codecs.open('../predict/predict.csv', 'w', 'utf-8')
csv_writer = csv.writer(file_predict)
csv_writer.writerow(['"uid"', '"score"'])
for i in xrange(len(uids)):
    csv_writer.writerow([uids[i], float(out[i])])
file_predict.close()
