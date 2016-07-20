import mnist
import mnist_loader
import numpy as np

training_data, validation_data, test_data =  mnist_loader.load_data_wrapper()
net = mnist.NetWork([784, 30, 10])
# net.SGD(training_data, 30, 10, 1,test_data=test_data)
import loaddata

training_data1=loaddata.LoadData(0,4000)
training_data=training_data[0:4000]
for x,y in zip(training_data,training_data1):
    print(np.argmax(x[0]),np.argmax(y[0]))
    
        