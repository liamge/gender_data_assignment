# label an argument response as sarcastic or nonsarcastic

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
import feature_extractor as fe
import numpy
import pickle
import sys
import csv
import xgboost as xgb
import pandas as pd
import matplotlib

datafile = "data/sarcasm_v2.csv"
conffile = sys.argv[1]

ndata = -1 # for testing feature extraction: optional arg to control how much of data to use. won't work for testing classification because it just takes the first n -- all one class
if len(sys.argv) > 2:
    ndata = int(sys.argv[2])

def load_data():
    with open(datafile) as f:
        return list(csv.reader(f))[0:ndata]

def load_conf_file():
    conf = set(line.strip() for line in open(conffile))
    return conf

def predict_sarcasm(X, Y):
    scores = cross_val_score(svm.SVC(), X, Y, scoring='accuracy', cv=10)
    return scores.mean()

def predict_XGB(X, Y):
    '''
    :param X: features
    :param Y: labels
    :return: accuracy of an Extreme Gradient Boosted model which is trained until no improvement is seen in 20 iters
    '''
    Y, _ = pd.factorize(Y)
    pickle.dump(X, open('features.pkl', 'wb'))
    pickle.dump(Y, open('labels.pkl', 'wb'))
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.333)
    preds, model = runXGB(X_train, Y_train, X_val, Y_val)
    scores = sum(preds == Y_val) / len(preds)
    return scores

def runXGB(train_X, train_y, test_X, test_y=None, seed_val=0, num_rounds=1000):
    '''
    :param train_X: train featureset
    :param train_y: train labels
    :param test_X: test featureset
    :param test_y: test labels
    :return: predictions of an Extreme Gradient Boosted model and the model itself
    '''
    param = {}
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain,'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

if __name__ == "__main__":
    data = load_data()
    conf = load_conf_file()
    features = fe.extract_features([line[-1] for line in data if line[0]=="GEN"], conf)
    labels = [line[1] for line in data if line[0]=="GEN"]

    print(predict_sarcasm(features, labels))
