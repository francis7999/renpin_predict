#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn import cross_validation, ensemble
import codecs

data_path = '../original_data'
train_X = pd.read_csv(os.path.join(data_path, 'train_x.csv'))
train_y = pd.read_csv(os.path.join(data_path, 'train_y.csv'))
Unlabeled_X_1 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_1.csv'))
Unlabeled_X_2 = pd.read_csv(os.path.join(data_path, 'train_unlabeled_2.csv'))
Unlabeled_X = pd.concat([Unlabeled_X_1, Unlabeled_X_2])
test_X = pd.read_csv(os.path.join(data_path, 'test_x.csv'))
