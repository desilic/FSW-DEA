import pandas as pd
import numpy as np

# data=pd.read_excel('llinear.xlsx')
#
#
#
# #1. 生成latex 格式
# print()
# for i in range(data.shape[0]):
#     print('A'+str(i+1),end=' ')
#     for j in range(data.shape[1]):
#         print('&'+str(data.iloc[i,j]),end=' ')
#     print()
# a=3

# 2. 生成excel

data=[]
for i in range(4):
    tmp=list(map(float,input().split(',')))
    data.append(tmp)
data=pd.DataFrame(data)
data.to_csv('TCGA-KIRC_evalution.csv')
print(data)
a=4