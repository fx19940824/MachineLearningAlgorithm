from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#plt.plot([1,2,3],[4,5,1])
#plt.show()
#fig=plt.figure()
#fig.suptitle('No axes on this figure')
#fig, ax_lst = plt.subplots(2, 2) 

'''a=pd.DataFrame(np.random.rand(4,5),columns=list('abcde'))
a_asndarray=a.values

x=np.linspace(0,2,100)
plt.plot(x,x,label='linear')
plt.plot(x,x**2,label='quadratic')
plt.plot(x,x**3,label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Simple Plot')
plt.legend()
plt.show()'''

'''x = np.arange(0, 10, 0.2)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()'''

'''plt.plot([1,2,3,4],[1,2,3,4])
plt.ylabel('some numbers')
plt.show()'''

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')