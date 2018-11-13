import numpy as np


def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def randCent(dataArr, k):
    n = np.shape(dataArr)[1]
    center = np.zeros((k, n))
    for j in range(n):
        minJ = min(dataArr[:, j])
        rangeJ = max(dataArr[:, j]) - minJ
        center[:, j] = minJ + rangeJ * np.random.rand(k)
    return center


def kmeans(dataArr, k, distMeas=distEclud, createCent=randCent):
    m, _ = np.shape(dataArr)
    centeroid = createCent(dataArr, k)
    clusterAssessment = np.zeros((m, 2))
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distIJ = distMeas(centeroid[j], dataArr[i])
                if distIJ < minDist:
                    minDist = distIJ
                    minIndex = j
            if clusterAssessment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssessment[i, :] = minIndex, minDist
        for cent in range(k):
            ptsInclust = dataArr[clusterAssessment[:, 0] == cent]
            centeroid[cent, :] = np.mean(ptsInclust,axis=0)
    return centeroid, clusterAssessment


if __name__ == '__main__':
    dataSet = np.array([[0.697, 0.460], [0.774, 0.376], [0.634, 0.264],
                        [0.608, 0.318], [0.556, 0.215], [0.403, 0.237],
                        [0.481, 0.149], [0.437, 0.211], [0.666, 0.091],
                        [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
                        [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
                        [0.593, 0.042], [0.719, 0.103], [0.359, 0.188],
                        [0.339, 0.241], [0.282, 0.257], [0.748, 0.232],
                        [0.714, 0.346], [0.483, 0.312], [0.478, 0.437],
                        [0.525, 0.369], [0.751, 0.489], [0.532, 0.472],
                        [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]])

    result = []
    for k in range(1, 4):
        centeroid, clusterAssessment = kmeans(dataSet, k)
        result.append([centeroid, clusterAssessment])
    print(result)