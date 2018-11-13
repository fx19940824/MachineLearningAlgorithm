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
    for labelDataSet in labelDataSets:
        m, n = np.shape(labelDataSet)
        m = float(m)
        pSet = []
        pSet.append((m + 1) / (dataSet.shape[0] + len(labelDataSets)))
        for i in range(6):
            trueSet = labelDataSet[labelDataSet[:, i] == test_data[i]]
            Ni = np.unique(dataSet[:, i]).shape[0]
            pSet.append((np.shape(trueSet)[0] + 1) / (m + Ni))
        for i in range(6, 8):
            u1 = np.mean(labelDataSet[:, i])
            a1 = np.var(labelDataSet[:, i])
            p1 = np.exp(-np.power(
                (test_data[i] - u1), 2) / (2 * a1)) / np.sqrt(2 * np.pi * a1)
            pSet.append(p1)
        res = 1.0
        for p in pSet:
            res *= p
        h[labelDataSet[0][-1]] = res

    maxP = 0
    for key in h.keys():
        if h[key] > maxP:
            maxP = h[key]
            resLabel = key
    print(resLabel)
