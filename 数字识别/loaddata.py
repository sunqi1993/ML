import struct
import numpy as np
from PIL import Image
import pickle
import platform
s=platform.system()
if s=='Windows':
    train_data_path=r"C:\Users\sunqi\Desktop\MNIST\train-images.idx3-ubyte"
    train_label_path=r"C:\Users\sunqi\Desktop\MNIST\train-labels.idx1-ubyte"
    path=r'C:\Users\sunqi\Desktop\MNIST\mnist.pkl.gz'
else:
    pass


buffer_data=open(train_data_path,'rb').read()
buffer_label=open(train_label_path,'rb').read()

def loadIMG():
    buff=buffer_data
    index=0
    magic,imgs,rows,cols=struct.unpack_from('>IIII',buff,index)
    index+=struct.calcsize('>IIII')
    for i in range(50):
        img=Image.new('L',(cols,rows))

        for j in range(cols):
            for k in range(rows): #行
                pix,=struct.unpack_from('>B',buff,index)
                index+=struct.calcsize('>B')
                img.putpixel((k,j),pix)
        label=Read_IMG_Label(i)[1]
        path=r'D:/新建文件夹/ML/数字识别/img/'+str(i)+'-'+str(label)+'.png'
        img.save(path)
    
def Read_IMG_Data(i):
    #读取第i张图片的数据
    index=16+i*28*28
    lenth=28*28
    s=struct.unpack_from('>'+str(lenth)+'B',buffer_data,index)
    #150为灰度阈值
    s=list(map(lambda x: x/255,s))
    data=np.array([s])
    return data.T

def Read_IMG_Label(i):
    index=i+8
    label,=struct.unpack_from('>B',buffer_label,index)
    y=np.zeros((10,1))
    y[label][0]=1
    return y,label

def LoadData(index,num):
    """ 导入数据 index:索引 num:导入数量
    return (x,y)型list
    """
    s=[]
    for k in range(num):
        s.append([Read_IMG_Data(index+k),Read_IMG_Label(index+k)[0]])
    return s

if __name__=='__main__':

    mk=Read_IMG_Data(0).reshape((28,28))
    loadIMG()
    k=Read_IMG_Label(0)


    import gzip
    f=gzip.open(path,"rb")
    a,b,c=pickle.load(f,encoding='bytes')
    s=a[0][0].reshape((28,28))
    print(len(a))
    # index=0
    # header=struct.unpack_from('>BBBBI',buffer_label,index)
    # print(header)
    # index+=struct.calcsize('>BBBBI')
    # print(index)
    # index=0
    # label=struct.unpack_from('>100B',buffer_label,index+8)
    # print(label)


        
