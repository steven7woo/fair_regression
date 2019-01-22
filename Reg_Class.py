import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
import data_parser as parser

def OLS_Reg(X, y):
    """
    Perform OLS Regression on a dataset
    No intercept here
    """
    reg_ols = sm.OLS(y, X).fit()
    return reg_ols.params


def L1_Reg(X, y):
    """
    Perform least absolute deviations regression We use quantile
    regression as a tool by setting the quantile value to be 0.5
    No intercept here

    """
    reg_l1 = QuantReg(y, X).fit(0.5)
    return reg_l1.params


def GD_L2_Reg(X, y):
    """
    A Gradient Descent Implementation for the L1 Regression Solver
    """
    X = tf.placeholder("float")
    y = tf.placeholder("float")
    d = np.shape(x)[1]
    W = tf.Variable(tf.random_uniform([d], -1.0, 1.0))
    y_hat = tf.multiply(X, W)
    loss = tf.reduce_mean(tf.square(y_hat - y))  # squared loss
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(2000):
        sess.run(train)
    print(step, sess.run(W))


def GD_L1_Reg(X, y):
    """
    A Gradient Descent Implementation for the L1 Regression Solver
    """
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    y_hat = W * X
    loss = tf.reduce_mean(tf.abs(y_hat - y))  # squared loss
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(2000):
        sess.run(train)
    print(step, sess.run(W))

# x, a, y = parser.clean_lawschool_gender(20)
# sige = 5
# x1 = np.random.randn(20, 10)
# OLS_Reg(x, y)
# L1_Reg(x, y)
# print(1)
