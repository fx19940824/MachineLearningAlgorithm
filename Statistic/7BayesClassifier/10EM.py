import numpy as np

def setHideFeature(dataSet,feature):
    dataSet[:,feature]=None

class BayesNetwork():

    class Node():
        def __init__(self,feature,value):
            self.feature=feature
            self.value=value
            self.child=[]
            self.parent=[]

        def setchild(self,child):
            self.child.append(child)

        def setparent(self,parent):
            self.parent.append(parent)

        

    def __init__(self,dataSet):
        self.dataSet=dataSet
        self.root=None
    
    def getB():
        visited=[]
        

    def BIC(self,dataSet):
        m=dataSet.shape[0]
        np.log(m)*

    def createNetwork(self,dataSet):
        m,n=np.shape(dataSet)
        self.root=
        for i in range(n-1):



if __name__ == '__main__':
    dataSet = np.array([['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 1],
                        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 1],
                        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 1],
                        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 1],
                        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 1],
                        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 1],
                        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 1],
                        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 1],
                        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0],
                        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0],
                        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0],
                        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0],
                        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0],
                        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0],
                        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0],
                        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0],
                        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0]])
    setHideFeature(dataSet,4)
    print(dataSet)

