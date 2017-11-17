#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import random as rnd
from copy import deepcopy

rnd.seed(0)
n_features = 100
importance_features = [4, 13, 29, 37, 43, 52, 68, 72, 81, 96]
n_pos_train_samples = 100
n_neg_train_samples = 9900
n_pos_valid_samples = 10
n_neg_valid_samples = 990
n_pos_test_samples = 10
n_neg_test_samples = 990
seed = 0

# 一部featureのみpositiveとnegativeで母数のrangeを変更
mu_samples = [[], []]
std_samples = [[], []]
mu_samples[0] = [rnd.randint(-3, 3) for _ in range(n_features)]
std_samples[0] = [rnd.randint(1, 3) for _ in range(n_features)]
mu_samples[1] = deepcopy(mu_samples[0])
std_samples[1] = deepcopy(std_samples[0])
for i in importance_features:
    mu = rnd.randint(-5, 5)
    std = rnd.randint(1, 5)
    while(mu == mu_samples[0][i]):
        mu = rnd.randint(-5, 5)
    while(std == std_samples[0][i]):
        std = rnd.randint(1, 5)
    mu_samples[1][i] = mu
    std_samples[1][i] = std

# 生成した母数を元にサンプルを作成
# train
train_samples = []
for i in range(n_pos_train_samples + n_neg_train_samples):
    sample = []
    label = 0
    if i >= n_neg_train_samples:
        label = 1
    sample.append(label)

    for f in range(n_features):
        sample.append(rnd.normal(mu_samples[label][f], std_samples[label][f]))
    train_samples.append(sample)
rnd.shuffle(train_samples)

# validation
valid_samples = []
for i in range(n_pos_valid_samples + n_neg_valid_samples):
    sample = []
    label = 0
    if i >= n_neg_valid_samples:
        label = 1
    sample.append(label)

    for f in range(n_features):
        sample.append(rnd.normal(mu_samples[label][f], std_samples[label][f]))
    valid_samples.append(sample)
rnd.shuffle(train_samples)

# test
test_samples = []
for i in range(n_pos_test_samples + n_neg_test_samples):
    sample = []
    label = 0
    if i >= n_neg_test_samples:
        label = 1
    sample.append(label)

    for f in range(n_features):
        sample.append(rnd.normal(mu_samples[label][f], std_samples[label][f]))
    test_samples.append(sample)
rnd.shuffle(test_samples)

# csvとlightgbm形式、libsvmで保存
# csv
columns = ['label']
for i in range(n_features):
    columns.append('feature_{}'.format(i))
pd.DataFrame(train_samples, columns=columns).to_csv('train.csv', index=False)
pd.DataFrame(valid_samples, columns=columns).to_csv('valid.csv', index=False)
pd.DataFrame(test_samples, columns=columns).to_csv('test.csv', index=False)

# lightgbm形式
with open('train.txt', 'wt') as f:
    for s in train_samples:
        print("\t".join(map(str, s)), file=f)

with open('valid.txt', 'wt') as f:
    for s in valid_samples:
        print("\t".join(map(str, s)), file=f)

with open('test.txt', 'wt') as f:
    for s in test_samples:
        print("\t".join(map(str, s)), file=f)

# libsvm
with open('train_libsvm.txt', 'wt') as f:
    for s in train_samples:
        print(s[0], " ".join(["{}:{}".format(i, si) for i, si in enumerate(s[1:])]), file=f)

with open('valid_libsvm.txt', 'wt') as f:
    for s in valid_samples:
        print(s[0], " ".join(["{}:{}".format(i, si) for i, si in enumerate(s[1:])]), file=f)

with open('test_libsvm.txt', 'wt') as f:
    for s in test_samples:
        print(s[0], " ".join(["{}:{}".format(i, si) for i, si in enumerate(s[1:])]), file=f)
