#实例 使用标准BP算法和累积性BP算法在西瓜数据集3.0上分别训练单隐层神经网络并对结果进行比较
'''
颜色：浅白 青绿 乌黑 1 2 3
根底：蜷缩 稍蜷 硬挺 1 2 3
敲声： 清脆 浊响 沉闷 1 2 3
纹理：模糊 稍微模糊 清晰 1 2 3
肚挤： 凹陷，稍凹 平摊 1 2 3
触感： 硬 软 1 2
'''
import numpy as np
import math
x=np.mat(  '2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1;\
            2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3;\
            3,3,3,3,3,3,2,3,2,3,1,1,2,2,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2;\
            1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1;\
            0.697,0.774,0.634,0.668,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719;\
            0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103\
            ').T
x=np.array(x)
xrow,xcol=x.shape
y=np.mat('1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0')
y=np.array(y).T

#定义隐层连接权值
v=np.random.random((10,8))
w=np.random.random((1,10))
#阈值
theta1=np.random.random((1,y.shape[1]))
theta0=np.random.random((1,10))

#学习率
alpha=0.01
i=1
#非线性函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
#改算法为累积性bp算法
while 1:
    i+=1
    b=sigmoid(np.dot(v,x.T)-np.tile(theta0.T,(1,xrow)))
    y_o=sigmoid(np.dot(w,b)-np.tile(theta1,(1,17))).T
    if abs(np.sum((y-y_o)**2))<0.05:
        print('v=%s\nw=%s\ntheta0=%s\ntheta1=%s\ny.T=%s\ny_o.T=%s\n' %(v,w,theta0,theta1,y.T,y_o.T))
        print(i)
        break
    g=y_o*(1-y_o)*(y-y_o)
    e=np.dot(g,w)*(b*(1-b)).T
    e=e.T  #10*17
    w+=alpha*np.dot(g.T,b.T)
    theta1-=alpha*sum(g)
    v+=alpha*np.dot(e,x)
    theta0-=alpha*e.sum(axis=1)



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              