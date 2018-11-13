import lightgbm as lgb
from sklearn import datasets

if __name__=='__main__':
    #add train data
    iris=datasets.load_iris()
    x_train,y_train=iris.data,iris.target

    train_data=lgb.Dataset(x_train,y_train)
