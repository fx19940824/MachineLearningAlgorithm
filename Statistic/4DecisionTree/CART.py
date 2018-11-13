import numpy as np
from sklearn import datasets

#class treeNode():
#    def __init__(self,feat,val,right,left):
#        featureToSplitOn=feat
#        valueOfSplit=val
#        rightBranch=right
#        leftBranch=left


class CartTree():
    def __init__(self, dataSet):
        self.dataSet = dataSet
        self.tree=None

    def binSplitDataSet(self, dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:, feature] > value), :][0]
        mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value), :][0]
        return mat0, mat1

    def __linearSolve(self,dataSet):
        m,n=np.shape(dataSet)
        X=np.ones((m,n))
        Y=np.ones((m,1))
        X[:,1:n]=dataSet[:,0,n-1]
        Y=dataSet[:,-1]
        xTx=X.T*X
        if np.linalg.det(xTx)==0:
            raise NameError('invert error')
        ws=xTx.I*(X.T*Y)
        return ws,X,Y
    
    def modelLeaf(self,dataSet):
        ws,X,Y=self.__linearSolve(dataSet)
        return ws
        
    def modelErr(self,dataSet):
        ws,X,Y=self.__linearSolve(dataSet)
        yHat=X*ws
        return sum(np.power(Y-yHat,2))

    def __regLeaf(self, dataSet):
        return np.mean(dataSet[:, -1])

    def __regErr(self, dataSet):
        return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

    def createTree(self, dataSet,leafType=__regLeaf,errType=__regErr, ops=(1, 4)):
        feat, val = self.chooseBestSplit(dataSet, ops)
        if feat == None:
            return val
        retTree = {}
        retTree['spInd'] = feat
        retTree['spVal'] = val
        lSet, rSet = self.binSplitDataSet(dataSet, feat, val)
        retTree['left'] = self.createTree(lSet, ops)
        retTree['right'] = self.createTree(rSet, ops)
        return retTree

    def chooseBestSplit(self, dataSet,leafType=__regLeaf,errType=__regErr, ops=(1, 4)):
        tolS = ops[0]
        tolN = ops[1]
        if dataSet[:, -1].max() == dataSet[:, -1].min():
            #if len(set(dataSet[:,-1].T.tolist()[0]))==1:
            return None, leafType(dataSet)
        m, n = np.shape(dataSet)
        S = errType(dataSet)
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(n - 1):
            for splitVal in set(dataSet[:, featIndex]):
                mat0, mat1 = self.binSplitDataSet(dataSet, featIndex, splitVal)
                if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
                    continue
                newS = errType(mat0) + errType(mat1)
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if S - bestS < tolS:
            return None, leafType(dataSet)
        mat0, mat1 = self.binSplitDataSet(dataSet, bestIndex, bestValue)
        if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
            return None, leafType(dataSet)
        return bestIndex, bestValue

    def __isTree(self, obj):
        return type(obj).__name__ == 'dict'

    def __getMean(self, tree):
        if self.__isTree(tree['right']):
            tree['right'] = self.__getMean(tree['right'])
        if self.__isTree(tree['left']):
            tree['left'] = self.__getMean(tree['left'])
        return (tree['left'] + tree['right']) / 2.0

    def prune(self, tree, testData):
        if np.shape(testData)[0] == 0:
            return self.__getMean(tree)
        if self.__isTree(tree['right']) or self.__isTree(tree['left']):
            lSet, rSet = self.binSplitDataSet(testData, tree['spInd'],
                                              tree['spVal'])
        if self.__isTree(tree['left']):
            self.prune(tree['left'],lSet)
        if self.__isTree(tree['right']):
            self.prune(tree['right'],rSet)
        if not self.__isTree(tree['left']) and not self.__isTree(tree['right']):
            lSet,rSet=self.binSplitDataSet(testData,tree['spInd'],tree['spVal'])
            errorNoMerge=sum(np.power(lSet[:,-1]-tree['left'],2))+sum(np.power(rSet[:,-1]-tree['right'],2))
            treeMean=(tree['left']+tree['right'])/2.0
            errorMerge=sum(np.power(testData[:,-1]-treeMean,2))
            if errorMerge<errorNoMerge:
                print('merging')
            else:
                return tree
        else:
            return tree



if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris['data']
    y = iris['target']
    dataSet = np.vstack([X.T, y]).T
    myTree = CartTree(dataSet)
    myTree.tree=myTree.createTree(dataSet,myTree.modelLeaf,myTree.modelErr)
    print(myTree.tree)
    myTree.prune(myTree.tree,dataSet)
    print(myTree.tree)
