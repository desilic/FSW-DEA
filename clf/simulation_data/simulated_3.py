import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

"""前38维"""

miu=0
sigma=1
data=np.random.normal(loc=miu,scale=sigma,size=400)#iid:miu,sigma相同的独立（正态）分布
for i in range(1,37): #37个iid的1维normal
    #生成1维normal随机数(1行500个)
    tmp=np.random.normal(loc=miu,scale=sigma,size=400)
    #拼接到data
    data=np.vstack([data,tmp])
#3.转置
data=data.T

"""后2维"""
miu1,sigma1=0.97, 0.4
miu2,sigma2=-1, 0.4
miu3=np.array([-0.04, 0.14])

cov=np.array([[1,0.997],[0.997,1]])

data1=np.random.normal(loc=miu1,scale=sigma1,size=200)
data2=np.random.normal(loc=miu2,scale=sigma2,size=200)

data3=np.random.multivariate_normal(miu3, cov, size=(200))
data4=np.random.multivariate_normal(miu3, cov, size=(200))

sig_var1=np.hstack([data1,data2])#合并仿真数据1的剩余2维的2个250行正态->1个500行正态
sig_var2=np.vstack([data3,data4])
data=np.vstack([data.T,sig_var1,sig_var2.T]).T  #将剩余2维拼接到前38维后


plt.plot(data1, data1,'.b')
plt.plot(data2, data2,'.r')
plt.show()
plt.plot(data3[:,0], data3[:,1],'.b')
plt.plot(data4[:,0], data4[:,1],'.r')
plt.show()

"""写到csv"""
# data=np.hstack([data,data3])      #将剩余2维拼接到前38维后

header=['Gene{}'.format(i) for i in range(data.shape[1])]
column=['Sample{}'.format(i) for i in range(data.shape[0])]
label=[1]*200+[0]*200


pd.DataFrame(data).to_csv('C:/Users/lenovo/Desktop/survival analysis/仿真数据/simulate_bbac.csv',index=column,header=header)