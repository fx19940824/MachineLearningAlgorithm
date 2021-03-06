import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

if __name__=='__main__':
    dataSet = np.array(
        [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1],
         [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
         [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0],
         [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
         [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
         [0.593, 0.042, 0], [0.719, 0.103, 0]])

    X=dataSet[:,0].reshape(-1,1)
    y=dataSet[:,1].reshape(-1,1)
    clf=SVR()
    clf=clf.fit(X,y)
    pred=clf.predict(X).reshape(-1,1)
    
    print(clf.support_vectors_)
    print(np.concatenate((pred,y),axis=1))