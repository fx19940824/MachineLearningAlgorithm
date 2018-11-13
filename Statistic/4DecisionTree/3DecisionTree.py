import numpy as np
import operator
from math import log


class decisionTree:
    def __init__(self, dataSet, labels):
        self.dataSet = dataSet
        self.classList = [example[-1] for example in dataSet]
        self.labels = labels
        self.tree = self.createTree(dataSet, labels)
        self.plotTree()

    def plotTree(self):
        curtree = [self.tree]
        while len(curtree) > 0:
            newtree = []
            print(curtree)
            for key in curtree.keys():
                if len(curtree[key]):
                    newtree.append(curtree[key])
            curtree = newtree

    def calcShannonEnt(self, dataSet):
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / len(dataSet)
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    def splitDataSet(self, dataSet, axis, value):
        result = []
        for featVec in dataSet:
            if featVec[axis] == value:
                curVec = featVec[:axis]
                curVec.extend(featVec[axis + 1:])
                result.append(curVec)
        return result

    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = 0
        bestFeature = -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = float(len(subDataSet)) / len(dataSet)
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(
            classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del (labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(
                self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree

    def createPrePruningTree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        del (labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)

        acc = self.validate(myTree, self.labels, self.dataSet, self.truthLabel)
        for value in uniqueVals:
            subLabels = labels[:]
            newTree = myTree
            newTree[bestFeatLabel][value] = self.createTree(
                self.splitDataSet(dataSet, bestFeat, value), subLabels)
            newacc = self.validate(newTree, self.labels, self.dataSet,
                                   self.truthLabel)
            if (newacc > myTree):
                myTree = newTree

        return myTree

    def findLeaf(self,inputTree):
        res=[]
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = self.classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
        return classLabel

    def createAfterPruningTree(self, dataSet, labels):
        self.tree = self.createTree(dataSet, labels)
        
        self.afterPruningTree
        
    def afterPruningTree(self,inputTree):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                if self.afterPruningTree(secondDict[key])==True:
                    acc=self.validate(self.tree,self.labels,self.dataSet,self.truthLabel)

            else:
                return True

        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':
                return False
        return True

    def classify(self, inputTree, featLabels, testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        return classLabel

    def validate(self, inputTree, featLabels, validationSet, truthLabel):
        validation = []
        for validata in validationSet:
            validation.append(
                self.classify(inputTree, featLabels, validationSet))
        if len(truthLabel) == len(validation):
            acc = 0.0
            for i in range(validation):
                if validation[i] == truthLabel[i]:
                    acc += 1
            acc = acc / len(validation)
            return acc
        else:
            return 0


if __name__ == '__main__':

    dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1],
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
               ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]]
    #color,root,sound,texture,region,touch,density,sugar,label=data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7],data[:,8]

    labels = [
        'color', 'root', 'sound', 'texture', 'region', 'touch', 'density',
        'sugar'
    ]
    for data in dataSet:
        del (data[7])
        del (data[6])
    del (labels[7])
    del (labels[6])
    MyTree = decisionTree(dataSet, labels)
    #print(MyTree.calContinuousEnt(density,label))
