#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn import cross_validation, ensemble
from sklearn.feature_selection import SelectFromModel
import codecs
from sklearn.externals import joblib

data_path = '../original_data'
Labeled_X = pd.read_csv(os.path.join(data_path, 'train_x.csv'))
Labeled_y = pd.read_csv(os.path.join(data_path, 'train_y.csv'))
Unlabeled_X_1 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_1.csv'))
Unlabeled_X_2 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_2.csv'))
#Unlabeled_X = pd.concat([Unlabeled_X_1, Unlabeled_X_2],join='inner',ignore_index=True)
Unlabeled_y_1 = pd.Series(-1 * np.ones((len(Unlabeled_X_1))))
Unlabeled_y_2 = pd.Series(-1 * np.ones((len(Unlabeled_X_2))))
del Unlabeled_X_1['uid']
del Unlabeled_X_2['uid']
del Labeled_X['uid']
del Labeled_y['uid']
a_Labeled_y = np.array(Labeled_y['y'])

GBRT = ensemble.GradientBoostingRegressor(n_estimators= 200, max_depth = 50, max_features= 'sqrt')
GBRT.fit(Labeled_X, a_Labeled_y)
joblib.dump(GBRT, 'GBRT.pkl')
