#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib


path = os.path.abspath(os.path.dirname(__file__))

test = pd.read_csv('{}/../data/test.csv'.format(path))
dtest = xgb.DMatrix(test.drop('label', axis=1), test['label'])
bst = xgb.Booster(model_file='{}/../model/dump.model'.format(path))

for p in bst.predict(dtest):
    print(p)
