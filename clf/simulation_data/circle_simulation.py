
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure(1)

# datasets.make_circles()专门用来生成圆圈形状的二维样本.factor表示里圈和外圈的距离之比.每圈共有n_samples/2个点，、

# 里圈代表一个类，外圈也代表一个类.noise表示有0.1的点是异常点
# plt.subplot(121)
x1, y1 = make_circles(n_samples=500, factor=0.5, noise=0.2)

color=['.r','.k']
# label_c=[color[i==1] for i in y1]

plt.title('make_circles function example')
#使用plot画散点图：分class画，1种class的点1个plot
plt.plot(x1[:, 0][y1==0], x1[:, 1][y1==0], '.r')#y1==0的点
plt.plot(x1[:, 0][y1==1], x1[:, 1][y1==1], '.k')#y1==1的点
#使用scatte散点图：可以1次画多个class的点，通过c=class或颜色 来区分不同class的点
# plt.scatter(x1[:, 0], x1[:, 1], c=label_c)#label_c==0的点1种颜色，label_c==1的点1种颜色

# plt.subplot(122)
# x1, y1 = make_moons(n_samples=1000, noise=0.1)
# plt.title('make_moons function example')
# plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)

plt.show()
