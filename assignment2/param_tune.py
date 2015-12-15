#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import json
import sys
import os
import data_utils.utils as du
import data_utils.ner as ner
from softmax_example import SoftmaxRegression
from nerwindow import WindowMLP
import itertools
from numpy import *
from multiprocessing import Pool
import random as rdm

random.seed(10)

wv, word_to_num, num_to_word = ner.load_wv('data/ner/vocab.txt',
                                           'data/ner/wordVectors.txt')

tagnames = ["O", "LOC", "MISC", "ORG", "PER"]
num_to_tag = dict(enumerate(tagnames))
tag_to_num = du.invert_dict(num_to_tag)

windowsize = 3
docs = du.load_dataset('data/ner/train')
X_train, y_train = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=windowsize)

docs = du.load_dataset('data/ner/dev')
X_dev, y_dev = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=windowsize)

docs = du.load_dataset('data/ner/test.masked')
X_test, y_test = du.docs_to_windows(docs, word_to_num, tag_to_num, wsize=windowsize)


nepoch = 5
N = nepoch * len(y_train)
k = 5 # minibatch size
schedules = ["epoch", "N", "mini_batch"]
sche_params = []
for sche_name in schedules:
    param = {"param": {"wv": wv, "windowsize":windowsize, "dims":[None, 100, 5], "reg":0.001, "alpha":0.01}, "setting_name": sche_name}
    sche_params.append(param)


def trainig_schedule(total_samples, num_train, k):
    rdm.seed(99)
    for i in xrange(total_samples / k):
        yield rdm.sample(xrange(num_train), k)

def experiment(param):
    pms = param["param"]
    sche_name = param["setting_name"]
    clf = WindowMLP(**pms)
    if sche_name == "epoch":
        schedule = itertools.chain(*itertools.repeat(xrange(len(y_train)), nepoch))
    elif sche_name == "N":
        schedule = random.randint(0, len(y_train), N)
    elif sche_name == "mini_batch":
        schedule = trainig_schedule(N, len(y_train), k)

    cost = clf.train_sgd(X_train, y_train, idxiter=schedule)
    result = {"cost": cost, "name": sche_name}
    return result

pool = Pool(processes=3)
results = pool.map(experiment, sche_params)
pool.close()
pool.join()

with open('result.json', 'w') as rj:
    json.dump(rj, results)

