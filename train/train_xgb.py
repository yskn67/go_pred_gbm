#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import xgboost as xgb


def parse_model(model, tree):
    nodeid = int(tree['nodeid'])
    del tree['nodeid']

    children = None
    if 'children' in tree:
        children = tree['children']
        del tree['children']

    if 'depth' in tree:
        del tree['depth']

    if 'split' in tree:
        tree['split'] = int(tree['split'].split('_')[1])

    model[nodeid] = tree

    if children is not None:
        for c in children:
            model = parse_model(model, c)

    return model

train = pd.read_csv("../data/train.csv")
valid = pd.read_csv("../data/valid.csv")

dtrain = xgb.DMatrix(train.drop('label', axis=1), train['label'])
dvalid = xgb.DMatrix(valid.drop('label', axis=1), valid['label'])

params = {'max_depth': 5,
          'learning_rate': 0.01,
          'n_estimators': 10,
          'min_child_weight': 10,
          'seed': 0}

bst = xgb.train(params,
                dtrain,
                evals=[(dvalid, 'logloss')]
                )

dump = bst.get_dump(dump_format='json')
dump_json = [json.loads(d.strip()) for d in dump]
with open('dump.json', 'wt') as f:
    json.dump(dump_json, f)
bst.dump_model('dump.txt')

dump_model = []
for d in dump_json:
    model = {}
    model = parse_model(model, d)

    model_list = [None] * len(model)
    for k, v in model.items():
        model_list[k] = v

    dump_model.append(model_list)

with open('dump_model.json', 'wt') as f:
    json.dump(dump_model, f)
