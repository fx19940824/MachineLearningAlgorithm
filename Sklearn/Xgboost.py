import xgboost as xgb
from sklearn import datasets
import sklearn.ensemble
import matplotlib
import pandas

if __name__ == '__main__':

    # add train data
    iris = datasets.load_iris()
    x_train, y_train = iris.data, iris.target

    dtrain = xgb.DMatrix(x_train, y_train)

    param = {'max_depth': 2}

    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(
        param, dtrain, num_round=10, evallist, early_stopping_rounds=10)

    xgb.plot_importance(bst)
