import numpy as np

def getLDA(x,y):
    u=[]
    for i in range(2):
        u.append(np.mean(x[y==i],axis=0))

    m,n=np.shape(x)
    Sw=0
    for i in range(x.shape[0]):
        xu=(x[i]-u[int(y[i])]).reshape(n,1)
        Sw+=np.dot(xu,xu.T)
    U,sigma,V=np.linalg.svd(Sw)
    Sw_inv=V.T*np.linalg.inv(np.diag(sigma))*U.T
    w=np.dot(Sw_inv,(u[0]-u[1]).reshape(n,1))
    return u,w

if __name__=='__main__':
    data=np.array([[0.697,0.460,1],
                    [0.774,0.376,1],
                    [0.634,0.264,1],
                    [0.608,0.318,1],
                    [0.556,0.215,1],
                    [0.403,0.237,1],
                    [0.481,0.149,1],
                    [0.437,0.211,1],
                    [0.666,0.091,0],
                    [0.243,0.267,0],
                    [0.245,0.057,0],
                    [0.343,0.099,0],
                    [0.639,0.161,0],
                    [0.657,0.198,0],
                    [0.360,0.370,0],
                    [0.593,0.042,0],
                    [0.719,0.103,0]]
                )
    density,sugar,label=data[:,0],data[:,1],data[:,2]

    x_train=np.c_[density,sugar]

    u,w=getLDA(x_train,label)

    center=[]
    for i in range(2):
        center.append(np.dot(w.T,u[i]))
    
    result=[]
    for i in range(x_train.shape[0]):
        x_test=np.dot(w.T,x_train[i].T)
        if np.abs(x_test-center[0])<np.abs(x_test-center[1]):
            result.append(0)
        else:
            result.append(1)
    print(result)
    

    
    