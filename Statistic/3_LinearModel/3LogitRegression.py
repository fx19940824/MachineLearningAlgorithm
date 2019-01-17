import numpy as np
import matplotlib.pyplot as plt

def likelihood(X,y,beta):
    sum=0
    m=X.shape[0]
    for i in range(m):
        bx=np.dot(beta,X[i].T)
        sum+=-y[i]*bx+np.math.log(1+np.math.exp(bx))
    return sum

def updatebeta(beta,x,y):
    dbeta=0
    ddbeta=0
    for i in range(x_train.shape[0]):
        bx=np.dot(beta,x[i].T)
        p1=1-1/(1+np.exp(bx))
        dbeta-=x[i].T*(y[i]-p1)
        ddbeta+=np.dot(x[i],x[i].T)*p1*(1-p1)

    beta=beta-(dbeta/ddbeta)
    return beta

import time
import progressbar

if __name__=='__main__':

    for i in progressbar.progressbar(range(100)):
        time.sleep(0.02)

    x_train=np.array([[0.697,0.460],
                    [0.774,0.376],
                    [0.634,0.264],
                    [0.608,0.318],
                    [0.556,0.215],
                    [0.403,0.237],
                    [0.481,0.149],
                    [0.437,0.211],
                    [0.666,0.091],
                    [0.243,0.267],
                    [0.245,0.057],
                    [0.343,0.099],
                    [0.639,0.161],
                    [0.657,0.198],
                    [0.360,0.370],
                    [0.593,0.042],
                    [0.719,0.103]]
                    )
    y_train=np.array([[1],[1],[1],[1],[1],[1],[1],[1],
                    [0],[0],[0],[0],[0],[0],[0],[0],[0]]
                    )
    
    max_iterator=500

    x_train=np.c_[x_train,np.ones(x_train.shape[0])]
    m,n=np.shape(x_train)

    delta=10
    loss=10
    iter=0
    beta=np.random.rand(1,3)
    while(delta>0.0001 and iter<max_iterator):
        
        curloss=likelihood(x_train,y_train,beta)
        delta=np.abs(loss-curloss)
        loss=curloss
        print('loss on step {} is {}'.format(iter,loss))
        beta=updatebeta(beta,x_train,y_train)
        iter+=1
        

    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    for i in range(m):
        if y_train[i]==0:
            plt.scatter(x_train[i][0],x_train[i][1],marker='o',color='k',s=100)
        if y_train[i]==1:
            plt.scatter(x_train[i][0],x_train[i][1],marker='o',color='g',s=100)
    plt.legend(loc='upper right')
    plt.show()