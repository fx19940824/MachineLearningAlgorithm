import os
import numpy as np


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

def LogisticRegression(x_train,y_train,threshold=0.0001,max_iterator=500):
    delta=10
    loss=10
    iter=0
    beta=np.random.rand(1,3)
    while(delta>threshold and iter<max_iterator):
        curloss=likelihood(x_train,y_train,beta)
        delta=np.abs(loss-curloss)
        loss=curloss
        print('loss on step {} is {}'.format(iter,loss))
        beta=updatebeta(beta,x_train,y_train)
        iter+=1

    return loss

if __name__=='__main__':
    data_dir=os.path.split(os.path.realpath(__file__))[0]
    data_dir=os.path.join(data_dir,'iris.data')
    f=open(data_dir,'r')
    data=f.read()
    rows=data.split('\n')
    full_data=[]
    for row in rows:
        split_row=row.split(",")
        full_data.append(split_row)

    full_data=np.array(full_data)

    