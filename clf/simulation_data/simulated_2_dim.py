import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

"""前38维"""

miu=random.uniform(10,30)
sigma=0.1
data=np.random.normal(loc=miu,scale=sigma,size=500)#iid:miu,sigma相同的独立（正态）分布
for i in range(1,38): #37个iid的1维normal
    # 1.先随机生成miu(5,15)，sigma(0,1)
    miu = random.uniform(10, 30)
    #2.再生成1维normal随机数(1行500个)
    tmp=np.random.normal(loc=miu,scale=sigma,size=500)
    #拼接到data
    data=np.vstack([data,tmp])
#3.转置
data=data.T

"""后2维"""
miu1=np.array([1,1])
miu2=np.array([1.11, 0.89])
# miu3=np.array([8.7,9.6])

cov1=np.array([[1,0.999],[0.999,1]])
# cov2=np.array([[2,-0.6],[-0.5,1.5]])

data1=np.random.multivariate_normal(miu1, cov1, size=(250))
data2=np.random.multivariate_normal(miu2, cov1, size=(250))
# data3=np.random.multivariate_normal(miu3, cov2, size=(500))

sig_var=np.vstack([data1,data2])#合并仿真数据1的剩余2维的2个250行正态->1个500行正态
data=np.hstack([data,sig_var])  #将剩余2维拼接到前38维后

plt.plot(data1[:,0], data1[:,1],'.b')
plt.plot(data2[:,0], data2[:,1],'.r')
plt.show()

"""写到csv"""
# data=np.hstack([data,data3])      #将剩余2维拼接到前38维后

header=['Gene{}'.format(i) for i in range(data.shape[1])]
column=['Sample{}'.format(i) for i in range(data.shape[0])]
label=[1]*250+[0]*250# p:1, n:0


pd.DataFrame(data).to_csv('C:/Users/lenovo/Desktop/survival analysis/仿真数据/simulate_ECFS-DEA.csv',index=column,header=header)