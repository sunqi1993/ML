#机器学习 感知机原始形式学习代码
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation 
#Data_set=np.array([[3,3,1],[4,3,1],[1,1,-1]])
Data_set=np.array([[3,3,1],[4,3,1],[1,1,-1],[1,2,1],[5,4,1],[4,1,-1],[6,2,-1],[0,3,1]])
#Data_set[1,0:2]
w=np.array([0,0])
b=0
theta=1
history=[]
def ganzhiji(dataset):

    global w,b,theta,history
    s=dataset.shape[0]
    history.append([w,b])
    for i in range(s):
        if(dataset[i,2]*(np.dot(w,dataset[i,0:2])+b)<=0):
            w=w+theta*dataset[i,2]*dataset[i,0:2]
            b=b+theta*dataset[i,2]
            ganzhiji(dataset)
            return w,b
            
if __name__=='__main__':
    s=ganzhiji(Data_set)
    print('最终迭代参数为：',s)
    
    fig=plt.figure()
    ax=plt.axes(xlim=(-6, 6), ylim=(-6, 6))
    line,=ax.plot([],[],'g',lw=2)
    label=ax.text([],[],'')
    
    #显示测试数据的开始框架
    def show():

        x, y, x_, y_= [],[],[],[]
        for i in range(Data_set.shape[0]):
            if(Data_set[i,2]>0):
                x.append(Data_set[i,0])
                y.append(Data_set[i,1])
            else:
                x_.append(Data_set[i,0])
                y_.append(Data_set[i,1])   
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.plot(x,y,'bo',x_,y_,'rx')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('感知机模型')
        return line,label

    
    #历史参数的动态显示
    def animate(i):
        w=history[i][0]
        b=history[i][1]
        if(w[1]==0):
           return line,label            
        x1=-6
        y1=-(b+w[0]*x1)/w[1]
        x2=6
        y2=-(b+w[0]*x2)/w[1]
        line.set_data([x1,x2],[y1,y2])
        x0=0
        y0=-(b+w[0]*x0)/w[1]
        str1=[w[0],w[1]],b
        label.set_text(str1)
        label.set_position([x0,y0])
        return line,label
    for i in history:
        print(i)
    #ImageMagick-7.0.1-Q16解码器需要提前装入才能生成动态视频
    plt.rcParams["animation.convert_path"]="C:\ImageMagick-7.0.1-Q16\magick.exe"
    anim=animation.FuncAnimation(fig,animate,init_func=show,frames=len(history),interval=500, repeat=True,blit=True)
    plt.show()
    anim.save('原始型.gif', fps=2, writer='imagemagick')
