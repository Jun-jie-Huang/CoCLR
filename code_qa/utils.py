# coding=utf-8
from __future__ import absolute_import, division, print_function

import csv
import json
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score, precision_score, recall_score

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    prec = precision_score(y_true=labels, y_pred=preds)
    reca = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": prec,
        "recall": reca,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "webquery":
        return acc_and_f1(preds, labels)
    if task_name == "staqc":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)

