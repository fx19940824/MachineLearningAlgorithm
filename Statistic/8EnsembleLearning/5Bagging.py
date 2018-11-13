import numpy as np
from sklearn import tree,ensemble
from sklearn.cross_validation import train_test_split
from sklearn import grid_search

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1
    return retArray


#dataArr: sample data (m*n)
#classLabels: class of each sample data (m*1)
#D:weight of each sample data(m*1)
def buildStump(dataArr, classLabels, D):
    m, n = np.shape(dataArr)
    minError = np.inf
    numSteps = 10
    bestStump = {}
    bestEst = None
    for i in range(n):
        rangeMin = dataArr[:, i].min()
        rangeMax = dataArr[:, i].max()
        step = (rangeMax - rangeMin) / numSteps
        for j in range(numSteps):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + step * j
                predict = stumpClassify(dataArr, i, threshVal, inequal)
                errArr = np.ones((m, 1))
                errArr[predict == classLabels] = 0
                weightedError = np.dot(D.T, errArr)
                if weightedError < minError:
                    minError = weightedError
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                    bestEst = predict

    return bestStump, minError, bestEst

def baggingDS(dataArr,numIt=50):
    classifierArr=[]
    X,y=dataArr[:,:-1],dataArr[:,-1]
    for i in range(numIt):
        X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.632)
        clf=tree.DecisionTreeClassifier()
        clf.fit(X_train,y_train)
        classifierArr.append(clf)
    return classifierArr

def evaluate(predict, classLabels):
    m = np.shape(predict)[0]
    error = np.ones((m, 1))
    error[predict == classLabels] = 0
    error = error.sum() / m
    return error

def baggingClassify(dataArr,classifierArr):
    X=dataArr[:,:-1]
    y=dataArr[:,-1]
    minError=np.inf
    bestPredict=None
    for clf in classifierArr:
        pred=clf.predict(X)
        error=evaluate(pred,y)
        if error<minError:
            minError=error
            bestPredict=pred
    return bestPredict,minError
            


if __name__=='__main__':
    dataSet = np.array(
        [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1],
         [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
         [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0],
         [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
         [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
         [0.593, 0.042, 0], [0.719, 0.103, 0]])

    classifiers=baggingDS(dataSet)
    pred,error=baggingClassify(dataSet,classifiers)