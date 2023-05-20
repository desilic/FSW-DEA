import numpy as np
import pandas as pd
import csv
from utils.tools import load_feature_impts_from_dir,get_gaussian_boundary
from sklearn import metrics
from Classify import Classifier, get_shuffled_data
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils.tools import load_data, Vimps, GraphVimps
import matplotlib.pyplot as plt
import warnings
import sys
warnings.filterwarnings('ignore')

#np.seterr(divide='ignore',invalid='ignore')

################################################4.clustering############################################################

#1.读取,处理gene_imps->vimps
vimp_path='E:/vimps_for_all_RNA_5feature/'
vimps=load_feature_impts_from_dir(vimp_path,True)
#2.画出高斯曲线（打印边界）
gauss_boundary=get_gaussian_boundary(vimps,10,True)#获取边界
#print(gauss_boundary)

#load_graph_impts_from_dir('E:/vimps_for_all_RNA_5feature/vimps_and_count多线程')
#std=load_and_show_feature_std(vimps,vimp_path)
#x=np.argsort(-vimps)#加-即可返回逆序下标

# im_feature_index=[]#重要特征的index--按重要性降序
# for i in np.argsort(-vimps):
#     #print(i,vimps[i])
#     if vimps[i]<gauss_boundary[3]:前9重要特征的index
#         break
#     im_feature_index.append(i)

########################################################################################################################
Data_path = 'E:/mi_RNAdata.csv'
Label_path ='E:/tcga_label.npy'

data = pd.read_csv(Data_path).iloc[:, 2:].T.values
label = np.load(Label_path)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3,random_state=1)

# #找显著特征对应gene名
# f=open('E:/mi_RNAdata.csv')
# genes=np.loadtxt(f,str,delimiter = ",",skiprows = 1)[:,1]
# names=[]
# for i in im_feature_index:
#     names.append(genes[i])
# print(names)

#分别计算前1/2/3个高斯的分类准确率
# guss_index=[0,1,8]#保存3个高斯min的下标
# for i in guss_index:
#     imf=im_feature_index[:i+1]
#     classifier = Classifier()
#     classifier = classifier.fit(x_train[:,imf], y_train)
#
#     idx, best_clf, best_score = classifier.find_best_ensembles(x_test[:,imf], y_test)
#     score= classifier.get_best_score(x_test[:,imf], y_test)
#     print(score)


################################### 5.incremental ########################################################################
""":使用前i重要gene，作为featur进行分类"""
scores=[]
best_classifier=[]
im_feature_index=np.argsort(-vimps)#全部特征（基因）
print("start")
#probas=[]
for i in range(1,len(im_feature_index)+1):
    imf=im_feature_index[:i]#得分最高的前i个特征

    classifier = Classifier()
    classifier = classifier.fit(x_train[:,imf], y_train)

    idx, best_clf, best_score = classifier.find_best_ensembles(x_test[:,imf], y_test)
    score= classifier.get_best_score(x_test[:,imf], y_test)

    #画前9个基因的roc曲线
    # proba=best_clf.predict_proba(x_test[:, imf])#返回每个样本（预测为0概率，预测为1概率）
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, proba[:,1], pos_label=1)
    # #print(fpr,tpr)
    # plt.plot(fpr,tpr,label='ROC {} curve'.format(i))
    # plt.xlim((0, 0.7))
    # plt.xlabel("False positive rate")
    # plt.ylabel("True positive rate")
    # plt.legend()
    #plt.show()

    #scores.append(score)
    best_classifier.append(idx)
    #probas.append(proba)

#show运行结果
#print(proba)
#print(scores);
print(best_classifier)
print('end')

#前i重要基因incremental的best_clf
plt.plot(range(1,len(best_classifier)+1),best_classifier,'.b')
plt.title('best classifier')
plt.xlabel("feature numbers");plt.ylabel("best_clf")
plt.show()

#前i重要基因incremental的得分scores
# plt.plot(range(1,len(scores)+1),scores)
# plt.xlabel("feature numbers");plt.ylabel("scores")
# plt.show()


