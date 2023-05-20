import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles



"""前48维"""

# miu=random.uniform(15,30)
# sigma=0.01
# data=np.random.normal(loc=miu,scale=sigma,size=500)#iid:miu,sigma相同的独立（正态）分布
# for i in range(1,48): #37个iid的1维normal
#     #生成1维normal随机数(1行500个)
#     miu = random.uniform(15, 30)
#     tmp=np.random.normal(loc=miu,scale=sigma,size=500)
#     #拼接到data
#     data=np.vstack([data,tmp])
# #3.转置
# data=data.T

data=pd.read_csv('simulate_circle_clf.csv')
"""后2维"""

# x1, y1 = make_circles(n_samples=500, factor=0.5, noise=0.2)
#
# x_m1=np.hstack([x1[:, 0][y1==1],x1[:, 0][y1==0]])
# x_m2=np.hstack([x1[:, 1][y1==1],x1[:, 1][y1==0]])
# tmp=np.vstack([x_m1,x_m2]).T

# plt.title('make_circles function example')
#使用plot画散点图：分class画，1种class的点1个plot
# plt.plot(x1[:, 0][y1==0], x1[:, 1][y1==0], '.r')#y1==0的点
# plt.plot(x1[:, 0][y1==1], x1[:, 1][y1==1], '.k')#y1==1的点

x1 = np.linspace(0, 250,250)
x2 = np.linspace(250, 500,250)
plt.plot(x1,data.iloc[250:, 5],'.r')#y==0
plt.plot(x2,data.iloc[:250, 5],'.k')#y==1
plt.xlabel('index')
plt.ylabel('expression')
plt.show()

plt.plot(x1,data.iloc[250:, 5],'.r')
plt.plot(x1,data.iloc[:250, 5],'.k')
plt.xlabel('index')
plt.ylabel('expression')
plt.title('one-dimention')
plt.show()



plt.plot(data.iloc[250:, 50], data.iloc[250:, 51], '.r')#y1==0的点
plt.plot(data.iloc[:250, 50], data.iloc[:250, 51], '.k')#y1==1的点
plt.xlabel('gene1')
plt.ylabel('gene2')
plt.title('non-linear')
plt.show()

"""写到csv"""
# data=np.hstack([data,data3])      #将剩余2维拼接到前38维后



label=pd.DataFrame([1]*250+[0]*250)#先转为DF 使其变成2维数组（n,1）而不是（n,）
label=np.array(label)#
data=np.hstack([label,data,tmp]) #将剩余2维拼接到前48维后
header=['label']+['Gene{}'.format(i) for i in range(data.shape[1]-1)]
column=['Sample{}'.format(i) for i in range(data.shape[0])]

pd.DataFrame(data).to_csv('C:/Users/lenovo/Desktop/survival analysis/仿真数据/simulate_circle_clf.csv',index=column,header=header)






