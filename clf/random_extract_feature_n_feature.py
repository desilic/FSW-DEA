import random
from clf.Classify import Classifier, get_shuffled_data #这里必须from clf.Classify 而不能from Classify 因为运行的工作路径在clf外
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import concurrent.futures as cf
from sklearn import preprocessing
from clf.utils.tools import load_data, Vimps, GraphVimps
from tqdm import tqdm 
import warnings 
import sys
warnings.filterwarnings('ignore')
'''
    > 1. 计算基因的重要性，没有交叉验证
'''

def train_and_trainscore(X, y):
    # ans 将迭代num_iter次的结果都存储起来，到结束后一起保存，节省计算机读写文件时间。
    # ans = np.zeros([num_iter, X.shape[1]+1])

    #2. 第二层划分

    #2.1先用70% training set 训练best classifier
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    classifier = Classifier()#实例化
    classifier = classifier.fit(x_train, y_train)#训练
    _, _, best_score = classifier.find_best_ensembles(x_test, y_test)#返回最好分类器的得分(acc 因为是1-err)

    # # 初始化每个特征的重要性， vimps[:-1] 记录每个特征重要性，vimps[-1]记录分类器的种类
    vimps = np.zeros((x_test.shape[1]))
    if best_score < 0.75:#意思是如果best分类器的得分<0.75,则认为抽中的特征都不重要？
        return 0, vimps

    #2.2再用剩余30% training set ：permutation 求imps
    # # 遍历所有特征(这里是108维)，计算
    for n_dim in range(x_test.shape[1]):
        # 打乱一维特征，再使用最好的分类器来预测一下结果
        shuffle_one_feature_data = get_shuffled_data(x_test, n_dim)
        score_shuffled = classifier.get_best_score(shuffle_one_feature_data, y_test, vote=False, accuracy=False)#best_clf+acc
        # 定义：该维特征的重要性 = 未打乱这个维度之前的准确性 - 打乱这个维度之后的准确性
        one_feature_imptns = best_score - score_shuffled# 因为这里是打乱前-打乱后，所以用的是arr
        # if one_feature_imptns < 0:
        #     one_feature_imptns = 0 
        #     print(n_dim)
        # 保存
        vimps[n_dim] = one_feature_imptns
    # # # vimps[-1] = idx
    return best_score, vimps #返回最好分类器的得分，和基因重要性

def extract_feature_for_acc(X, y, n_features, times, save_dir):
    # get n features from X randomly. 从所有特征中随机抽取n_features个特征
    sample = range(X.shape[1])
    vimps_ans = Vimps(X.shape[1], save_dir)#一个用来保存显著特征的对象？
    #graph_vimps = GraphVimps(X.shape[1], save_dir)
    feature_dict = {}
    feature_count = {}

    for _ in range(times):
        #extract_features:1个list 存随机抽取的特征的编号
        extract_features = random.sample(sample, n_features)#从sample中随机抽取n_features个元素返回list
        # print(extract_features)
        # extract_features = np.array(extract_features)
        # # get those n features's score. NOT SHUFFLE
        #对所有的样本，只训练抽取的特征
        best_score, vimps = train_and_trainscore(X[:,extract_features], y)
        #vimps 是抽取的20个特征的显著程度，best_score是对应特征下的最好得分

        # vimps = train_and_trainscore(X, y)
        # feature score. SHUFFLE.
        # for i, vimps in zip(extract_features, vimps):
        #     if i in feature_dict:
        #         feature_dict[i] += vimps
        #     else:
        #         feature_dict[i] = vimps
        # feature_dict = {i:vimp for i, vimp in zip(extract_features, vimps)}

        #把(特征的编号:显著程度)放入dict---feature_dict
        #第i个特征的抽中次数---feature_dict[i]
        for i, vimp in zip(extract_features, vimps):#zip:将a,b对应元素打包成元组，元组构成的list
            if i in feature_dict:
                feature_dict[i] += vimp
                feature_count[i] += 1
            else:
                feature_dict[i] = vimp
                feature_count[i] = 1

    vimps_ans.update(feature_dict, feature_count)
    # if best_score != 0:
    #     graph_vimps.update(feature_dict, best_score)
    vimps_ans.save_vimps()
    #graph_vimps.save_graph()
    print('End:', time.ctime())


if __name__ == '__main__':

    # name = sys.argv[1]
    name = 'KIRC'
    WORK_PATH = r'..'
    Data_path = os.path.join(WORK_PATH, 'E:/mi_RNAdata.csv')
    Label_path = os.path.join(WORK_PATH, 'E:/tcga_label.npy')
    vimps_save_dir = os.path.join(WORK_PATH, 'E:/vimps_for_all_RNA_5feature')

    print('Start!')
    print(time.ctime())

    data = pd.read_csv(Data_path).iloc[:, 2:].T.values
    label = np.load(Label_path)


    #1. 第1层样本划分
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data, label, test_size=0.5)

    # not_zero_index = df.iloc[:, 2:].std(axis=1) != 0
    # t = df.loc[not_zero_index,:]
    # t.to_csv(r'D:\BBD_DATA\TCGA_DATA\TCGA-LIHC\exp_processed1.csv')
    # x_train, x_test, y_train, y_test = train_test_split(data, label, stratify=label, test_size=0.5, random_state=1)
    # data = scipy.stats.zscore(data)
    # Training set and Testing Set

    if not os.path.exists(vimps_save_dir):
        os.makedirs(vimps_save_dir)



####################单进程############################################
    # n_feature 随机抽n个基因， times: 随机抽多少次， save_dir: 基因重要性的保存文件夹路径
    #extract_feature_for_acc(data_x_train, data_y_train, n_features=20, times=10, save_dir=vimps_save_dir)
    # extract_feature_for_acc(data, label, n_features=20, times=10, save_dir=vimps_save_dir)
    # vimps = Vimps(num_of_features=data.shape[1])
    # result = extract_feature_for_acc(data, label, n_feature)
    # vimps.update(result)
    # vimps.save_vimps()
    # plt.plot(vimps.avg_vimps())
    # plt.show()
    
####################多进程############################################
    #n_feature = 5/20 100次/200次/500次
    test_times=100#100/200/500

    works = 100
    n_feature = 20
    times_for_a_work = int(data.shape[1] * (test_times / n_feature ) / works) #1424个基因，1次抽5个，需要抽1424/5次（来让每个基因都抽一次）。每个基因抽100次（*100），每个进程平均要run几次（/works）

    if os.cpu_count() == 16:
        n_jobs = 10
    elif os.cpu_count() == 12:
        n_jobs = 10
    else:
        n_jobs= int(os.cpu_count() * 0.8 // 2 * 2)
    jobs = []
    #extract_feature_for_acc(x_train, y_train, n_feature, times_for_a_work, graph_save_dir)
    print('*' * 20)
    with cf.ProcessPoolExecutor(max_workers=12) as pool:
        for i in range(works):
            jobs.append(pool.submit(extract_feature_for_acc, data, label, n_feature, times_for_a_work, vimps_save_dir))

    print('Compute the importances of feature completed!')
    print(time.ctime())