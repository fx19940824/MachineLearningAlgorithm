import numpy as np


def convert(label_name):
    return {
        '青绿': 1,
        '乌黑': 2,
        '浅白': 3,
        '蜷缩': 1,
        '稍蜷': 2,
        '硬挺': 3,
        '浊响': 1,
        '沉闷': 2,
        '清脆': 3,
        '清晰': 1,
        '稍糊': 2,
        '模糊': 3,
        '凹陷': 1,
        '稍凹': 2,
        '平坦': 3,
        '硬滑': 1,
        '软粘': 2
    }.get(label_name)


def splitDataSet(dataSet, feature):
    uniques = np.unique(dataSet[:, feature])
    spDataSet = []
    for value in uniques:
        spDataSet.append(dataSet[dataSet[:, feature] == value])
    return spDataSet


if __name__ == '__main__':
    dataSet = np.array([['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1],
                        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, 1],
                        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1],
                        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1],
                        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, 1],
                        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1],
                        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1],
                        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1],
                        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0],
                        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0],
                        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0],
                        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0],
                        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0],
                        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0],
                        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, 0],
                        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, 0],
                        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]])
    for example in dataSet:
        for i in range(6):
            example[i] = convert(example[i])
    dataSet = dataSet.astype(np.float)

    test_data = np.array(['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460])
    for i in range(6):
        test_data[i] = convert(test_data[i])
    test_data = test_data.astype(np.float)

    h = {}
    labelDataSets = splitDataSet(dataSet, -1)
    D = dataSet.shape[0]
    N = len(labelDataSets)
    for labelDataSet in labelDataSets:
        resP = 0
        for i in range(6):
            xiSet = labelDataSet[labelDataSet[:, i] == test_data[i]]
            Dcxi = xiSet.shape[0]
            Ni = np.unique(dataSet[:, i]).shape[0]
            pcx = (Dcxi + 1) / (D + N * Ni)
            for j in range(6):
                xijSet = xiSet[xiSet[:, j] == test_data[j]]
                Dcxij = xijSet.shape[0]
                Nj = np.unique(dataSet[:, j]).shape[0]
                p = (Dcxij + 1) / (Dcxi + Nj)
                pcx *= p
            for j in range(6, 8):
                u = np.mean(xiSet[:, j])
                a = np.var(xiSet[:, j])
                p = np.exp(-np.power(
                    (test_data[j] - u), 2) / (2 * a)) / np.sqrt(2 * np.pi * a)
                pcx *= p
            resP += pcx
        for i in range(6, 8):
            xiSet = labelDataSet
            u = np.mean(xiSet[:, i])
            a = np.var(xiSet[:, i])
            pcx = np.exp(-np.power(
                (test_data[i] - u), 2) / (2 * a)) / np.sqrt(2 * np.pi * a)
            for j in range(6):
                xijSet = xiSet[xiSet[:, j] == test_data[j]]
                Dcxij = xijSet.shape[0]
                Nj = np.unique(dataSet[:, j]).shape[0]
                p = (Dcxij + 1) / (Dcxi + Nj)
                pcx *= p
            for j in range(6, 8):
                u = np.mean(xiSet[:, j])
                a = np.var(xiSet[:, j])
                p = np.exp(-np.power(
                    (test_data[j] - u), 2) / (2 * a)) / np.sqrt(2 * np.pi * a)
                pcx *= p
            resP += pcx
        h[labelDataSet[0][-1]] = resP

    maxP = 0
    for key in h.keys():
        if h[key] > maxP:
            maxP = h[key]
            resLabel = key
    print(resLabel)
