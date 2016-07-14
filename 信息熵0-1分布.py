import math as m
import numpy as np
from matplotlib import pyplot as plt
x=np.linspace(0,1,1001)  
#去掉开始的0和结尾的1
x=x[1:-1]
f=lambda x:-x*m.log2(x)-(1-x)*m.log2(1-x)
y=list(map(f,x))
plt.plot(x,y)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.xlabel('x')
plt.ylabel('y')
plt.title('0-1分布下的信息熵随概率变化范围')
plt.show()