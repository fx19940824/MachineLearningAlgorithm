import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error

if __name__=='__main__':
    dataSet = np.array(
        [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1],
         [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
         [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0],
         [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
         [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
         [0.593, 0.042, 0], [0.719, 0.103, 0]])

    X,y=dataSet[:,:-1],dataSet[:,-1]
    
    params={'n_estimators':500,'max_depth':4,'min_samples_split':2,'learning_rate':0.01,'loss':'ls'}

    clf=ensemble.GradientBoostingRegressor(**params)

    clf.fit(X,y)
    mse=mean_squared_error(y,clf.predict(X))
    print('MSE: {0}'.format(mse))
