import struct
from PIL import Image
import numpy as np
path=r'C:\Users\sunqi\Desktop\MNIST\train-images.idx3-ubyte'

def loadIMG():
    f=open(path,'rb')
    buff=f.read()
    index=0
    magic,imgs,rows,cols=struct.unpack_from('>IIII',buff,index)
    index+=struct.calcsize('>IIII')
    print(magic,imgs,rows,cols)

    for i in range(10):
        img=Image.new('L',(cols,rows))

        for j in range(cols):
            for k in range(rows):
                pix,=struct.unpack_from('>B',buff,index)
                index+=struct.calcsize('>B')
                img.putpixel((k,j),pix)
        img.save('./数字识别/img/'+str(i)+'.png')

class  NetWork(object):
    def __init__(self,sizes):
        self.num_layer=len(sizes)
        self.size=sizes
        self.biases=[np.random.randn(k,1) for k in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    
    #非线性S函数
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    #前相通路计算输出
    def feedfoward(self,a):
        k=[a]
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
            k.append(a)
        return a,k
    
    #误差逆向传播算法 计算出梯度
    def backprop(self,x,y):
        list_w=[np.zeros(w.shape) for w in self.weights]
        list_b=[np.zeros(b.shape) for b in self.biases]

        activation=x
        activations=[x] 
        zs=[]
        dw,db=[],[]
        #计算前向输出
        for b,w in zip(self.biases,self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=sigmoid(z)
            activations.append(activation)

        lenth=len(zs)
        #求出逆向误差传播梯度值
        for i in range(lenth).__reversed__():
            if i==lenth-1:delta=y*(1-activations[i+1])*activations[i+1]
            #delta_l=w_(l+1).transpose() X delta_l+1 *a_l*(1-a_l)
            else:delta=np.dot(self.weights[i].T,delta)*(1-activations[i+1])*activations[i+1]
            delta_w=delta*activations[i].T
            delta_b=delta
            dw.insert(0,delta_w)
            db.insert(0,delta_b)
        return db,dw
        


    def update_mini_batch(self,mini_batch,eta):
        #对于一小组数据进行梯度下降 eta是学习速率
        param_w=[np.zeros(w.shape) for w in self.weights]
        param_b=[np.zeros(b.shape) for b in self.biases]
        
        for x,y in mini_batch:
            #大部分算法在该逆向传播函数上
            delta_b,delta_w=self.backprop(x,y)
            nb=[a+b for a,b in zip(param_b,delta_b)]
            nw=[a+b for a,b in zip(param_w,delta_w)]

            self.weights=[w-eta/len(mini_batch)*dw for w,dw in zip(self.weights,nw)]
            self.biases=[b-eta/len(mini_batch)*db for b,db  in zip(self.biases,nb)]

            
        

    #随机梯度下降算法
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        
        if test_data:n_test=len(test_data)
        n=len(training_data)
        for j in xrange(epochs):
            #将数组内容全部打乱
            np.random.shuffle(test_data)
            mini_batches=[test_data[k:k+mini_batch_size] 
                          for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                #算法待更新
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                #如果有测试数据打印测试结果
                print('Epoch{0},{1} / {2}'.format(
                    j,self.evaluate(test_data),n))
            else:
                print('Epoch {0} complete'.format(j))


s=NetWork([3,4,1])

print(1)
# loadIMG()