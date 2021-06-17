import torch
from IPython import display
from matplotlib import pyplot as plt
import  numpy as np
import random


input_nums = 2    #特征数量
num_examples = 1000    #样本个数
true_w = [2,-3.4]    #真实数据
true_b = 4.2
features = torch.randn(num_examples,input_nums,dtype=torch.float32)    #随机生成样本
labels = true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b    #用真实数据给随机数据做标签
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)


w = torch.tensor(np.random.normal(0, 0.01, (input_nums, 1)), dtype=torch.float32)   #随机生成可训练的参数
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)   #设置参数可以用梯度来迭代
b.requires_grad_(requires_grad=True)


def data_iter(batch_size, features, labels):      #数据加载器
    num_examples = len(features)     #样本数量
    indices = list(range(num_examples))     #产生样本数量的下标
    random.shuffle(indices)      #打乱样本下标
    for i in range(0,num_examples,batch_size):      #随机取数据
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
    yield features.index_select(0,j),labels.index_select(0,j)   #产生随机取的数据


def linearReg(X, w, b):      #设计训练模型
    return torch.mm(X, w)+b


def Squared_loss(y_hat, y):     #损失函数
    return (y_hat-y.view(y_hat.size()))**2/2  #让y的大小与Y_hat一致


def sgd(params, lr, batch_size):    #参数优化算法
    for param in params:
        param.data -= lr*param.grad/batch_size


lr = 0.03
num_epochs = 100
net = linearReg
loss = Squared_loss
batch_size = 10

for epoch in range(num_epochs):     #训练过程
    for X,y in data_iter(batch_size,features,labels):
        l =loss(net(X,w,b),y).sum()    #一个batch的损失，根据这个损失来进行反向传播
        l.backward()
        sgd([w,b],lr,batch_size)

        w.grad.data.zero_()   #下一次反反向传播之前，要把梯度置为0
        b.grad.data.zero_()
    train_l = loss(net(features,w,b), labels)     #用更新一次后的w，b来计算总损失
    print('epoch %d,loss %f'%(epoch+1,train_l.mean().item()))


print(true_w,'\n',w)
print(true_b,'\n',b)

