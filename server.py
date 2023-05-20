from random import randrange

from flask import Flask, render_template,session,request
from flask import Response, abort


# from pyecharts import options as opts
# from pyecharts.charts import Bar

from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename

import clf
import numpy as np
import pandas as pd
import csv
from clf.utils.tools import load_feature_impts_from_dir,get_gaussian_boundary
from sklearn import metrics
from clf.Classify import Classifier, get_shuffled_data,ensemble_predict_by_voted,ensemble_predict_proba
from sklearn.model_selection import train_test_split, StratifiedKFold
from clf.utils.tools import load_data, Vimps, GraphVimps
import matplotlib.pyplot as plt
import warnings
import sys
warnings.filterwarnings('ignore')

import random
from clf.Classify import Classifier, get_shuffled_data ,Error_score#这里必须from clf.Classify 而不能from Classify 因为运行的工作路径在clf外
import os, time
from sklearn import metrics
from sklearn.metrics import auc
import concurrent.futures as cf
from sklearn import preprocessing

from tqdm import tqdm
from clf.random_extract_feature_n_feature import train_and_trainscore, extract_feature_for_acc

from sklearn.ensemble import VotingClassifier

UPLOAD_FOLDER = 'uploads' #上传文件要储存的目录
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xlsx','xls'} #允许上传的文件扩展名的集合

app = Flask(__name__) #加上 , static_folder="templates" 就打不开img，但是pyecharts貌似要加上 我试试去掉pyecharts还是可以用
app.config['SECRET_KEY']=os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["DOWNLOAD_FOLDER"]='downloads'



@app.route('/', methods=['GET'])
def index():
    return render_template('indexx.html', uid=clf.utils.smuid.short_uuid())
@app.route('/help', methods=['GET'])
def get_help():
    return render_template('help.html')


def show_figure(pid):#替换test.py的第一块code
    # 1.读取,处理gene_imps->vimps
    # vimp_path = 'static/vimps_for_all_RNA_n_feature/'
    # pid='muaKIL8P'
    vimp_path = 'static/vimps_for_all_RNA_n_feature/' + pid +'/'
    vimps = load_feature_impts_from_dir(vimp_path, True)
    # 2.画出高斯曲线（打印边界）
    gauss_boundary = get_gaussian_boundary(vimps, 10, True)  # 获取边界

    # 3. 求gauss—boundary score阈值的 对应index
    x = np.argsort(-vimps)
    guss_boundary_index = []

    i, j = 0, len(gauss_boundary) - 1
    while i < len(x) and j >= 0:
        if vimps[x[i]] < gauss_boundary[j]:
            guss_boundary_index.append(i) #i多一位，下面[:i]取不到i
            j -= 1
        i += 1

    # gauss阈值对应的index
    guss_index = guss_boundary_index[:4]  # 数据不同，guass阈值个数不同，统一只显示前4个
    guss_index.append(len(x))

    return vimps,guss_index

def running_clf(n,r,pid):
    """
        paramet：n，r
        先导入random_extract_feature_n_feature.py的2个函数

    :return:
    """



    print('Start!')
    print(time.ctime())

    # data = pd.read_csv(Data_path).iloc[:, 2:].T.values
    # label = np.load(Label_path)

    # pid='muaKIL8P'
    File_path = 'uploads/' + pid + '.csv'
    vimps_save_dir = 'static/vimps_for_all_RNA_n_feature/' + pid + '/'

    file= pd.read_csv(File_path)
    data = file.iloc[:, 2:].values
    label = file.iloc[:, 1].values



    # 1. 第1层样本划分
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data, label, test_size=0.5)


    # Training set and Testing Set

    if not os.path.exists(vimps_save_dir):
        os.makedirs(vimps_save_dir)

    #extract_feature_for_acc(data_x_train, data_y_train, n_features=20, times=10, save_dir=vimps_save_dir)
####################多进程############################################
    #n_feature = 5/20 100次/200次/500次
    test_times= r #100#100/200/500

    works = 100
    n_feature = n #5
    times_for_a_work = int(data.shape[1] * (test_times / n_feature ) / works)
    #1424个基因，1次抽5个，需要抽1424/5次（来让每个基因都抽一次）。每个基因抽100次（*100），每个进程平均要run几次（/works）

    if os.cpu_count() == 16:
        n_jobs = 10
    elif os.cpu_count() == 12:
        n_jobs = 10
    else:
        n_jobs= int(os.cpu_count() * 0.8 // 2 * 2)
    jobs = []
    #print("cpu_count:{},n_jobs:{}".format(os.cpu_count(),n_jobs))
    #extract_feature_for_acc(x_train, y_train, n_feature, times_for_a_work, graph_save_dir)
    print('*' * 20)
    with cf.ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for i in range(works):
            jobs.append(pool.submit(extract_feature_for_acc, data_x_train, data_y_train, n_feature, times_for_a_work, vimps_save_dir))

    print('Compute the importances of feature completed!')
    print(time.ctime())

    #return data, label
    return data_x_test,  data_y_test

def running_ruc(vimps, guss_index, data_x,  data_y, times, pid):
    # probas = []
    # scores = []
    metris = []
    im_feature_index = np.argsort(-vimps)  # 全部特征（基因）
    print("start roc")

    print('times:{}'.format(times))
    print('guss:{}'.format(len(guss_index)))
    print(guss_index)
    print(time.ctime())


    # for i in [guss_index[-1]]:
    for i in guss_index:
        imf = im_feature_index[:i]  # 得分最高的前i个gauss内的所有特征

        # 3.training、testing and measuring
        # 构建集成分类器
        cur_dim_clf=[]#当前维度下的r个base_clf
        # max_best_clf,max_best_score=0,0
        for k in range(times):
            x_train, x_test, y_train, y_test = train_test_split(data_x, data_y)
            classifier = Classifier()
            classifier = classifier.fit(x_train[:, imf], y_train)  # 训练base clf

            idx, best_clf, best_score = classifier.find_best_ensembles(x_test[:, imf], y_test)
            # if k==0:
            #     max_best_clf,max_best_score=best_clf, best_score
            # if max_best_score < best_score:
            #     max_best_score=best_score
            #     max_best_clf=best_clf
            cur_dim_clf.append(best_clf)
        print('emsemble training complete '+str(i))
        print(time.ctime())

        y_pred = ensemble_predict_by_voted(cur_dim_clf, x_test[:, imf])  # 根据论文best_clf即LR：有98%的准确率，改成vote则准确率会降低
        #print(y_pred)
        #print(y_test)
        Accuracy, err_metrics = Error_score(y_test, y_pred)
        # print("Accuracy: {}".format(Accuracy))
        err_metrics["Accuracy"] = round(Accuracy, 3)
        err_metrics["features"] = len(imf)

    


        # 画前i个gauss的roc曲线
        # proba = max_best_clf.predict_proba(x_test[:, imf])
        # proba = best_clf.predict_proba(x_test[:, imf])  # 返回每个样本（预测为0概率，预测为1概率）
        proba = ensemble_predict_proba(cur_dim_clf, x_test[:, imf])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, proba[:, 1], pos_label=1)
        # print(fpr,tpr)
        AUC = auc(fpr, tpr)
        # plt.plot(fpr, tpr, label='ROC {} curve'.format(guss_index.index(i) + 1))
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {})'.format(round(AUC, 3)))
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlim((0, 0.7))
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        # plt.legend(loc="lower right")
        # plt.title('ensemble')
        fig.savefig(r'static/img/roc_{}.png'.format(guss_index.index(i) + 1),
                    dpi=400, bbox_inches='tight')
        #plt.show()

        # scores.append(score)
        # probas.append(proba)
        metris.append(err_metrics)
        # print(metris)

    # show运行结果
    # print(proba)
    # print(scores);
    # print(metris)
    print(time.ctime())
    print('end roc')

    writer = pd.ExcelWriter('downloads/metrics/{}_{}.xlsx'.format("evaluation",pid), engine='openpyxl')
    #writer = pd.ExcelWriter('downloads/metrics/{}.xlsx'.format("xxx"), engine='openpyxl')

    # File_path = 'uploads/' + pid + '.csv'
    # file = pd.read_csv(File_path)
    # name=pd.DataFrame(file.columns[2:])

    df1 = pd.DataFrame(vimps)
    # df1 = pd.concat([name, vimps], axis=1)
    df2=pd.DataFrame(metris)
    df1.to_excel(writer, "gene importance")
    df2.to_excel(writer, "metrics")

    writer.save()
    writer.close()
    # pd.DataFrame(metris).to_csv('downloads/metrics/{}.csv'.format("xxx"),index=0)
    return metris


@app.route('/upload', methods=['GET', 'POST'])#上传参数
def upload():
    # 输入参数：
    times = request.form.get("times")
    n_feature = request.form.get("n_feature")
    pid=request.form.get("page_uid")
    #
    # if int(n_feature)<0:
    #     resp = Response("login fail")
    #     abort(resp)

    # 在这里调用clf的code？用函数代替test.py模块，因为test.py本身就是在调用classify，tools
    X,Y=running_clf(int(n_feature), int(times), str(pid))
    # 图片 直接保存到当前项 （具体：flask项目的stadic目录）
    vimps,guss_index=show_figure(str(pid))
    # 数据 则函数返回到当前函数
    metris=running_ruc(vimps, guss_index, X,Y, int(times),str(pid))
    print("before final page")

    return render_template("indexx.html", times=times, n_feature=n_feature, metris=metris, pid=str(pid))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#上传文件，把用户重定向到已上传文件的 URL
@app.route('/upload_file', methods=['GET', 'POST'])#上传文件
def upload_file():
    #print('已经进入后端flask/upload_file!!!')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename) #上传时的文件名
            filename = request.form.get("uuid") + '.' + file.filename.split('.')[-1]   #自定义文件名+保留原后缀名
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # #输入参数：
            # times = request.form.get("times")
            # n_feature=request.form.get("n_feature")

            #在这里调用clf的code？用函数代替test.py模块，因为test.py本身就是在调用classify，tools
            #running_clf()
            #图片 直接保存到当前项 （具体：flask项目的stadic目录）
            # show_figure()

            #数据 则函数返回到当前函数
            print("**************")
            session["name"] = filename#"dd" #设置会话级参数name，实现点击超链接跳转到 下载页面

            File_path = 'uploads/' + request.form.get("uuid") + '.csv'
            data = pd.read_csv(File_path).iloc[:, 2:].values
            session["feature_numbers"]=data.shape[1]
            #显示到当前页面
            print(filename,data.shape[1])
            return str(data.shape[1])
                # render_template("1111.html", feature_numbers=data.shape[1])

    return render_template("1111.html")


#为已上传的文件提供服务，使之能够被用户下载。
from flask import send_from_directory
# 定义download_file 视图来为上传文件夹中的文件提供服务，
@app.route('/downloads/<name>')
def download_file(name):
    directory = "downloads"
    if name[-4:]=='xlsx':
        directory="downloads/metrics"
    return send_from_directory(directory, name, as_attachment=True)
    #return send_from_directory("downloads/metrics", name, as_attachment=True)
    # 发现：加上as_attachment=True，才能直接下载到本地，而不是缓存到浏览器打开
    # # url_for("download_file", name=name) 依据文件名生成下载 URL  所以下载使用这个命令就行
    # return redirect(url_for('download_file', name=filename))

@app.route('/try', methods=['POST'])
def try_up():
    return render_template("indexx.html")

if __name__ == "__main__":
    #app.run()
    app.run(host="0.0.0.0",port=80)#50008787 host="0.0.0.0",port=5000
