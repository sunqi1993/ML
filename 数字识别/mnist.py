import numpy as np
import random
from loaddata import *

class  NetWork(object):
    def __init__(self,sizes):
        self.num_layer=len(sizes)
        self.size=sizes
        self.biases=[np.random.randn(k,1) for k in sizes[1:]]
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    
    #前相通路计算输出
    def feedfoward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid(np.dot(w,a)+b)
        return a
    
    #误差逆向传播算法 计算出梯度
    def backprop(self,x,y):
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
        delta=0
        #求出逆向误差传播梯度值
        for i in reversed(range(lenth)):
            if i==lenth-1:delta=(activations[i+1]-y)*df_sigmpoid(activations[i+1])
            #delta_l=w_(l+1).transpose() X delta_l+1 *a_l*(1-a_l)
            else:delta=np.dot(self.weights[i+1].T,delta)*df_sigmpoid(activations[i+1])
            delta_w=np.dot(delta,activations[i].T)
            delta_b=delta
            dw.append(delta_w)
            db.append(delta_b)
        dw.reverse()
        db.reverse()
        return db,dw
        
    def update_mini_batch(self,mini_batch,eta):
        #对于一小组数据进行梯度下降 eta是学习速率
        nw=[np.zeros(w.shape) for w in self.weights]
        nb=[np.zeros(b.shape) for b in self.biases]
        
        for x,y in mini_batch:
            #大部分算法在该逆向传播函数上
            delta_b,delta_w=self.backprop(x,y)
            #计算梯度累加和
            nb=[a+b for a,b in zip(nb,delta_b)]
            nw=[a+b for a,b in zip(nw,delta_w)]

        self.weights=[w-eta/len(mini_batch)*dw for w,dw in zip(self.weights,nw)]
        self.biases=[b-eta/len(mini_batch)*db for b,db  in zip(self.biases,nb)]
        
    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedfoward(x)),np.argmax(y)) for x,y in test_data]
        return sum([int(x==y) for x,y in test_results])        



    #随机梯度下降算法
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        
        if test_data:n_test=len(test_data)
        n=len(training_data)
        for j in range(epochs):
            #将数组内容全部打乱
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                #算法待更新
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                #如果有测试数据打印测试结果
                print('Epoch{0},{1} / {2}'.format(j,self.evaluate(test_data),n_test))
            else:
                print('Epoch {0} complete'.format(j))

    # def backprop1(self, x, y):
    #     """Return a tuple ``(nabla_b, nabla_w)`` representing the
    #     gradient for the cost function C_x.  ``nabla_b`` and
    #     ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    #     to ``self.biases`` and ``self.weights``."""
    #     nabla_b = [np.zeros(b.shape) for b in self.biases]
    #     nabla_w = [np.zeros(w.shape) for w in self.weights]
    #     # feedforward
    #     activation = x
    #     activations = [x] # list to store all the activations, layer by layer
    #     zs = [] # list to store all the z vectors, layer by layer
    #     for b, w in zip(self.biases, self.weights):
    #         z = np.dot(w, activation)+b
    #         zs.append(z)
    #         activation = sigmoid(z)
    #         activations.append(activation)
    #     # backward pass
    #     delta = self.cost_derivative(activations[-1], y) * \
    #         self.sigmoid_prime(zs[-1])
    #     nabla_b[-1] = delta
    #     nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    #     # Note that the variable l in the loop below is used a little
    #     # differently to the notation in Chapter 2 of the book.  Here,
    #     # l = 1 means the last layer of neurons, l = 2 is the
    #     # second-last layer, and so on.  It's a renumbering of the
    #     # scheme in the book, used here to take advantage of the fact
    #     # that Python can use negative indices in lists.
    #     for l in range(2, self.num_layer):
    #         z = zs[-l]
    #         sp = self.sigmoid_prime(z)
    #         delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
    #         nabla_b[-l] = delta
    #         nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    #     db1,dw1=self.backprop1(x,y)
    #     s=[]
    #     for x,y in zip(dw1,nabla_w):
    #         s.append(int((x==y).all()))
    #         k=x-y
    #     if sum(s)==2:
    #         print('相等')
    #     else:
    #         print('不等{0}'.format(s))

    #     return (nabla_b, nabla_w)
    
    # def sigmoid_prime(self,x):
    #     return sigmoid(x)*(1-sigmoid(x))


    # def cost_derivative(self, output_activations, y):

    #     """Return the vector of partial derivatives \partial C_x /
    #     \partial a for the output activations."""
    #     return (output_activations-y)
#非线性S函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

def df_sigmpoid(x):
    return x*(1-x)  

if __name__=='__main__':
    s=NetWork([28*28,30,10])
    train_data=LoadData(0,40000)
    test_data=LoadData(40000,10000)
    s.SGD(train_data,300,1,1,test_data=test_data)
    
   