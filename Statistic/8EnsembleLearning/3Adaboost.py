import numpy as np
import operator

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


def adaBoostTrainDS(dataArr, numIt=50):
    classLabels = dataArr[:, -1]
    classLabels[classLabels == 0] = -1
    dataArr = dataArr[:, :-1]
    m, n = np.shape(dataArr)
    classLabels = classLabels.reshape(m, 1)
    D = np.ones((m, 1)) / m
    weakClassArr = []
    aggClassEst = np.zeros((m, 1))
    for i in range(numIt):
        bestStump, minError, bestEst = buildStump(dataArr, classLabels, D)
        alpha = 0.5 * np.log((1 - minError) / max(minError, 1e-16))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply((-alpha * classLabels), bestEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * bestEst
        aggErrors = np.ones((m, 1))
        aggErrors[np.sign(aggClassEst) != classLabels] = 0
        aggErrors = aggErrors.sum() / m
        if aggErrors == 0:
            break
    return weakClassArr


def adaClassify(dataArr, classifierArr):
    m, n = np.shape(dataArr)
    aggClassEst = np.zeros((m, 1))
    for classifier in classifierArr:
        predict = stumpClassify(dataArr, classifier['dim'],
                                classifier['thresh'], classifier['ineq'])
        aggClassEst += classifier['alpha'] * predict
    return np.sign(aggClassEst)


def evaluate(predict, classLabels):
    m = np.shape(predict)[0]
    error = np.ones((m, 1))
    error[predict == classLabels] = 0
    error = error.sum() / m
    return error


if __name__ == '__main__':
    dataSet = np.array(
        [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1],
         [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
         [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0],
         [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
         [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
         [0.593, 0.042, 0], [0.719, 0.103, 0]])
    weakClassArr = adaBoostTrainDS(dataSet)
    print(weakClassArr)
    classLabels = dataSet[:, -1].reshape(dataSet.shape[0], 1)
    predict = adaClassify(dataSet[:, :-1], weakClassArr)
    error = evaluate(predict, classLabels)
    print(error)
