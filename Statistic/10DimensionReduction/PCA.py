import numpy as np
from PIL import Image
import os
import glob
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pcaAlg(dataMat, topNfeat=999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVects = np.linalg.eig(covMat)
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = np.dot(meanRemoved, redEigVects)
    reconMat = lowDDataMat * redEigVects.T + meanVals
    return lowDDataMat, reconMat


def plotRatio(variance_ratio):
    plt.figure()
    plt.plot(variance_ratio, 'k', linewidth=2)
    plt.xlabel('n_components', fontsize=16)
    plt.ylabel('ratio', fontsize=16)
    plt.show()


if __name__ == '__main__':
    data_dir = os.path.split(os.path.realpath(__file__))[0]
    file_dir = os.path.join(data_dir, 'yalefaces/')
    file_list = glob.glob(file_dir + '/*')

    h = 243
    w = 320

    X = []
    for file_name in file_list:
        x = np.array(Image.open(file_name))
        X.append(x.reshape(x.size))
    X_data = np.array(X)

    pca = PCA(n_components=50)
    pca.fit(X_data)
    plotRatio(pca.explained_variance_ratio_)

    #reverse
    mu = np.mean(X_data, axis=0)
    for nComp in [1, 5, 10, 15, 20]:
        time_start = time.time()
        Xhat = np.dot(
            pca.transform(X_data)[:, :nComp], pca.components_[:nComp, :])
        Xhat += mu
        for img in Xhat:
            img = img.reshape((h, w))
            plt.imshow(img)
            plt.show()
        time_end = time.time()

    print(time_end - time_start)
    '''dataSet = np.array(
        [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1],
         [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
         [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0],
         [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
         [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
         [0.593, 0.042, 0], [0.719, 0.103, 0]])

    X,y=dataSet[:,:-1],dataSet[:,-1]
    lowDMat,reconMat=pca(X,1)'''
