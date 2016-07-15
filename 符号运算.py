import numpy as np
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ]) #input
y = np.array([[0,1,1,0]]).T #output
syn0 = 2*np.random.random((3,4)) - 1  #v权值
syn1 = 2*np.random.random((4,1)) - 1  #w权值
for j in range(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0)))) #输出b 
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1)))) #输出y
    l2_delta = (y - l2)*(l2*(1-l2)) #g行向量
    
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    
    syn1 += l1.T.dot(l2_delta)
    if j==1:print(l1,'\n',l2_delta)
    syn0 += X.T.dot(l1_delta)