import numpy as np
from keras.datasets import mnist

inputArr1 = np.array([[1, 1], [7, 7]])
inputArr2 = np.array([[1, 1, 1, 0, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0],
                      [5, 5, 5, 0, 0], [1, 1, 0, 2, 2], [0, 0, 0, 3, 3],
                      [0, 0, 0, 1, 1]])
inputArr3 = np.array([[4, 4, 0, 2, 2], [4, 0, 0, 3, 3], [4, 0, 0, 1, 1],
                      [1, 1, 1, 2, 0], [2, 2, 2, 0, 0], [1, 1, 1, 0, 0],
                      [5, 5, 5, 0, 0]])

U, sigma, VT = np.linalg.svd(inputArr2)


def euclidSim(inA, inB):
    return 1.0 / (1 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(np.dot(inA.T, inB))
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    m, n = np.shape(dataMat)
    simTotal = 0.0
    ratSimtotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = np.nonzero(
            np.logical_and(dataMat[:, item] != 0, dataMat[:, j] != 0))[0]
        if len(overLap) == 0:
            continue
        similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        ratSimtotal += similarity * userRating
    return 0 if simTotal == 0 else ratSimtotal / simTotal


def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.eye(4) * Sigma[:4]
    xformedItems = np.dot(dataMat.T, np.dot(U[:, :4], Sig4))
    simTotal = 0.0
    ratSimtotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :], xformedItems[j, :])
        simTotal += similarity
        ratSimtotal += similarity * userRating
    return 0 if simTotal == 0 else ratSimtotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :] == 0)[0]
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append([item, estimatedScore])
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


#inA, inB = inputArr2[:, 0], inputArr2[:, 0]

#coe1, coe2, coe3 = euclidSim(inA, inB), pearsSim(inA, inB), cosSim(inA, inB)
#print(coe1, coe2, coe3)

#result = recommend(inputArr3, 2, estMethod=svdEst)
#print(result)


def imgCompress(input_img):
    if len(input_img) < 3:
        return [np.linalg.svd(input_img)]
    result = []
    for img in input_img:
        U, Sigma, VT = np.linalg.svd(img)
        result.append({'U':U, 'Sigma':Sigma, 'VT':VT})
    return result


def imgDeCompress(input_code, numSV=3):
    output_img = []
    for code in input_code:
        sigRecon = code['Sigma'][:numSV]
        sigRecon = np.eye(numSV) * sigRecon
        reconMat = np.dot(
            np.dot(code['U'][:, :numSV], sigRecon), code['VT'][:numSV, :])
        output_img.append(reconMat)
    return np.array(output_img)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
compresscode = imgCompress(x_train[:5])
out_img = imgDeCompress(compresscode)
