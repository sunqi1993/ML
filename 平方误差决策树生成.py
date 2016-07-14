import numpy as np
import math
data=np.array([[1,4.5],[2,4.75],[3,4.91],[4,5.34],[5,5.8],[6,7.05],
                [7,7.9],[8,8.23],[9,8.7],[10,9]])
y=data[:,1]

#判断列表内的所有类型是否都是数值型切长度不为1
def allint(listdata):
    s=len(listdata)
    k=0
    for i in listdata:
        if(isinstance(i,list)!=True):k=k+1
    if(k==s and s!=1):
        return True
    else:
        return False

#将一个内部全int型list分割为两部分
def seprate(listdata):
    lenth=len(listdata)
    y=np.array(listdata)
    s=range(lenth)[1:]
    k=[]
    for i in s:
        ave1=sum(y[0:i])/i
        ave2=sum(y[i:])/(lenth-i)
        # print(y[0:i],y[i:])
        # print(ave1,ave2)
        value=sum((y[0:i]-ave1)**2)+sum((y[i:]-ave2)**2)
        k.append(value)
    valuelist=np.array(k)
    index=valuelist.argsort()[0] #分割点序号
    listdata=[listdata[0:index+1],listdata[index+1:]]
    return index,listdata
    

y=y.tolist()

class q2tree(object):
    def __init__(self,listdata):
        self.value=listdata
        if(allint(listdata)):
            self.index=seprate(listdata)[0]
            self.lchild=q2tree(seprate(listdata)[1][0])
            self.rchild=q2tree(seprate(listdata)[1][1])
s1=q2tree(y)
#s1为树形结构 展开后为各个分支
print(1)