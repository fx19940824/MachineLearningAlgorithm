import numpy as np
import operator

def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def kNN(inX,dataArr,k,distMeas=distEclud):
    X,y=dataArr[:,:-1],dataArr[:,-1]
    m,n=np.shape(X)
    
    dist=np.zeros((m))
    for i in range(m):
        dist[i]=distMeas(X[i],inX)
        #dist[i,:]=i,distMeas(X[i],inX)
    index=dist.argsort()
    classLabels={}
    for i in range(k):
        voteLabel=y[index[i]]
        classLabels[voteLabel]=classLabels.get(voteLabel,0)+1
    sortClassCount=sorted(classLabels.items(),key=operator.itemgetter(1),reverse=True)
    return sortClassCount[0][0]


if __name__=='__main__':
    dataSet = np.array(
        [[0.697, 0.460, 1], [0.774, 0.376, 1], [0.634, 0.264, 1],
         [0.608, 0.318, 1], [0.556, 0.215, 1], [0.403, 0.237, 1],
         [0.481, 0.149, 1], [0.437, 0.211, 1], [0.666, 0.091, 0],
         [0.243, 0.267, 0], [0.245, 0.057, 0], [0.343, 0.099, 0],
         [0.639, 0.161, 0], [0.657, 0.198, 0], [0.360, 0.370, 0],
         [0.593, 0.042, 0], [0.719, 0.103, 0]])

    m,n=np.shape(dataSet)
    X,y=dataSet[:,:-1],dataSet[:,-1]
    pred=[]
    for i in range(m):
        pred.append(kNN(X[i],dataSet,3))
    for i in range(m):
        print([y[i],pred[i]])