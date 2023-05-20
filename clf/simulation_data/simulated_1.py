import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

"""前47维"""

# miu=random.uniform(15,30)
# sigma=0.01
# data=np.random.normal(loc=miu,scale=sigma,size=500)#iid:miu,sigma相同的独立（正态）分布
# for i in range(1,47): #37个iid的1维normal
#     #生成1维normal随机数(1行500个)
#     miu = random.uniform(15, 30)
#     tmp=np.random.normal(loc=miu,scale=sigma,size=500)
#     #拼接到data
#     data=np.vstack([data,tmp])
# #3.转置
# data=data.T
#
# """后3维"""
# miu1,sigma1=0.94, 0.3
# miu2,sigma2=1.4, 0.3 #1.7的太大，score>0.4了
# miu3=np.array([1, 1])
# miu4=np.array([1.09,0.91])#两个变量的miu差别越大，样本点分的越开 #重要性略低
# cov=np.array([[1,0.998],[0.998,1]])
#
# data1=np.random.normal(loc=miu1,scale=sigma1,size=250)
# data2=np.random.normal(loc=miu2,scale=sigma2,size=250)
#
# data3=np.random.multivariate_normal(miu3, cov, size=(250))
# data4=np.random.multivariate_normal(miu4, cov, size=(250))
#
# sig_var1=np.hstack([data1,data2])#合并仿真数据1的剩余2维的2个250行正态->1个500行正态
# sig_var2=np.vstack([data3,data4])
# data=np.vstack([data.T,sig_var1,sig_var2.T]).T  #将剩余2维拼接到前38维后



# x1 = np.linspace(0, 250,250)
# x2 = np.linspace(250, 500,250)

# plt.plot(data1,'.r')
# plt.plot(data2,'.k')
# plt.xlabel('index')
# plt.ylabel('expression')
# plt.show()
#
# x1 = np.linspace(0, 250,250)
# x2 = np.linspace(250, 500,250)
# plt.plot(x1,data1,'.r')
# plt.plot(x2,data2,'.k')
# plt.xlabel('index')
# plt.ylabel('expression')
# plt.show()
#
#
# plt.plot(data3[:,0], data3[:,1],'.r')
# plt.plot(data4[:,0], data4[:,1],'.k')
# plt.xlabel('gene1')
# plt.ylabel('gene2')
# plt.show()

#二次画图
data=pd.read_csv('simulate_web_clf.csv')

x1 = np.linspace(0, 250,250)

plt.plot(x1,data.iloc[250:, 49],'.r')
plt.plot(x1,data.iloc[:250, 49],'.k')
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

header=['label']+['Gene{}'.format(i) for i in range(data.shape[1])]
column=['Sample{}'.format(i) for i in range(data.shape[0])]
label=pd.DataFrame([1]*250+[0]*250)#先转为DF 使其变成2维数组（n,1）而不是（n,）
label=np.array(label)#
data=np.hstack([label,data]) #将剩余2维拼接到前38维后


pd.DataFrame(data).to_csv('C:/Users/lenovo/Desktop/survival analysis/仿真数据/simulate_web_clf.csv',index=column,header=header)