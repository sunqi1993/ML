import numpy as np
from matplotlib import pyplot as plt
import math
path=r'D:\足迹\编程记录\Python\相关书籍\机器学习实战-附加源码\MLiA_SourceCode\machinelearninginaction\Ch05\testSet.txt'
data=np.loadtxt(path,dtype=float)

#定义连续sigmoid函数 模拟阶跃函数
sigmoid=lambda x: 1/(1+math.exp(-x))


def trainal(dataset):
    data=dataset[:,0:2]
    label=dataset[:,2]
    row,col=data.shape
    
    alpha=0.0001  #学习速率
    MaxCycle=500    
    theta=np.ones((col,1))
    theta1=0
    time=0
    while 1:
        s=np.dot(data,theta)
        h=list(map(sigmoid,s))
        h=np.array(h).reshape((row,1))
        error=label.reshape(row,1)-h
        theta1=theta
        theta=theta+alpha*np.dot(data.T,error)

        s=(theta-theta1)/theta
        k=s[:,0]
        lenk,count=len(k),0
        time=time+1
        for i in k:
            print(theta.tolist())
            if abs(i)<1E-4:count=count+1
        if lenk==count:
            break
    print(time,'\n',theta)
trainal(data)
