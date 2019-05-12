"""
Take the regression result and threshold the scores
to obtain binary predictions
"""

from __future__ import print_function
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import eval as evaluate
import solvers as solvers
import exp_grad as fairlearn
import run_exp as run_exp
from scipy.stats import norm
import data_parser as parser
print = functools.partial(print, flush=True)
import data_parser as parser
from load_logged_exp import *




DATA_SPLIT_SEED = 4
full = False


x, a, y = parser.clean_adult_full()
if not full:
    x, a, y = run_exp.subsample(x, a, y, 2000)


def score2pred(reg_res, thresh=0.5):
    train_eval = reg_res['train_eval']
    test_eval = reg_res['test_eval']
    
    eps_vals = train_eval.keys()

    class_train_pred = {}
    class_test_pred = {}
    train_err = {}
    test_err = {}

    x_train, a_train,  y_train, x_test, a_test, y_test = run_exp.train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)

    



    print(eps_vals)
    for eps in eps_vals:
        class_train_pred[eps] = 1 *  (train_eval[eps]['pred'] >= thresh) 
        class_test_pred[eps] = 1 * (test_eval[eps]['pred'] >= thresh) 
        
        train_err[eps] = sum(abs(y_train - class_train_pred[eps])) / len(a_train)


        print(class_train_pred[eps])

        







score2pred(adult_short_SVM)
