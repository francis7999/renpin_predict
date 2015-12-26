#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn import cross_validation, ensemble
import codecs

data_path = '../original_data'
Labeled_X = pd.read_csv(os.path.join(data_path, 'train_x.csv'))
Labeled_y = pd.read_csv(os.path.join(data_path, 'train_y.csv'))
Unlabeled_X_1 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_1.csv'))
Unlabeled_X_2 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_2.csv'))
Unlabeled_X = pd.concat([Unlabeled_X_1, Unlabeled_X_2],join='inner',ignore_index=True)
Unlabeled_y = pd.Series(np.zeros((len(Unlabeled_X))))
del Unlabeled_X['uid']
del Labeled_X['uid']
del Labeled_y['uid']
del Unlabeled_X_1
del Unlabeled_X_2

X = np.array(Unlabeled_X)







test_X = pd.read_csv(os.path.join(data_path, 'test_x.csv'))

