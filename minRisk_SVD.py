# coding=utf-8

import matplotlib
import QUANTAXIS as QA
import pandas as pd
import numpy as np
# np.linalg.svd()
import sklearn.decomposition as skd
import matplotlib.pyplot as plt

# infos = QA.QA_fetch_stock_list_adv()
# result = pd.merge(stock_diff,infos.loc[codes]['name'], left_index=True,right_index=True)

codes = QA.QA_fetch_stock_block_adv().get_block(['生物医药','化学制药']).code
data =QA.QA_fetch_stock_day_adv(codes[:10],'2017-01-05','2017-03-25').to_hfq()

data =QA.QA_fetch_stock_day_adv(['002415','601155','000735','300558'],'2017-01-05','2017-12-25').to_hfq()

data.price_chg.values
data.price_chg[:, '000735'].values
data.price_chg['2017-01-10']
data.data.loc["2017-01-03","000735"]

data.price_chg.unstack().values.T

###################
# lgR = np.log(data.close/data.pre_close)
# lgR.plot()
# lgR_mat = np.delete(lgR.unstack().values,0,axis=0)
# lgR_mat.shape
#
# n_components = 4
#
# meanVal=np.mean(lgR_mat,axis=0) #按列求均值，即求各个特征的均值
# meanlgR_mat=lgR_mat-meanVal
# covMat = np.cov(meanlgR_mat,rowvar=0) #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
#
# eigVals,eigVects=np.linalg.eig(np.mat(covMat)) #求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
#
# #argsort将x中的元素从小到大排列，提取其对应的index(索引)
# eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
# #print(eigValIndice)
# n_eigValIndice=eigValIndice[-1:-(n_components+1):-1]   #最大的n个特征值的下标
# n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
# lowDDataMat=meanlgR_mat*n_eigVect               #低维特征空间的数据
# reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #重构数据
# pd.DataFrame(lowDDataMat).plot(x=0,y=1,kind='scatter')

###################
lgR = np.log(data.close/data.pre_close)
lgR_us = lgR.unstack()
lgR_us.dropna(0, inplace= True)
lgR_mat = lgR_us

# ret = data.pct_change
# lgR = ret.apply(lambda x : np.log(x+1))
# lgR_us = lgR.unstack()
# lgR_us.dropna(0, inplace= True)
# lgR_mat = lgR_us
# covMat = np.cov(lgR_mat,rowvar=0) #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

corMat = np.mat(lgR_mat.interpolate().corr())
eigVals,eigVects=np.linalg.eig(corMat) #求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量

lowDDataMat=corMat*n_eigVect               #低维特征空间的数据
reconMat=(lowDDataMat*n_eigVect.T)  #重构数据
pd.DataFrame(lowDDataMat).plot(x=0,y=1,kind='scatter')


# Definition of the Marchenko-Pastur density
#http://www.doc88.com/p-9877831718045.html
def marchenko_pastur_pdf(x, lanbda, sigma=1):
    b=np.power(sigma*sigma*(1 + np.sqrt(lanbda)),2) # Largest eigenvalue
    a=np.power(sigma*sigma*(1 - np.sqrt(lanbda)),2)# Smallest eigenvalue
    return (1/(2*np.pi*sigma*sigma*lanbda*x))*np.sqrt((b-x)*(x-a))*(0 if (x > b or x <a ) else 1)

N,T = lgR_mat.shape
lanbda = T/N
sigma = 1

max_theoretical_eval = np.power(sigma*sigma*(1 + np.sqrt(lanbda)),2)

# Filter the eigenvalues out
eigVals[eigVals <= max_theoretical_eval] = 0

filtered_matrix = np.dot(eigVects, np.dot(np.diag(eigVals), eigVects.T))


f = plt.figure()
ax = plt.subplot(121)
ax.imshow(corMat)
plt.title("Original")
ax = plt.subplot(122)
plt.title("Filtered")
ax.imshow(filtered_matrix)
a = ax.imshow(filtered_matrix)
cbar = f.colorbar(a, ticks=[-1, 0, 1])


def show_compare_plot(eigVals, lanbda, sigma):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    ax.hist(eigVals, normed=True, bins=10) # Histogram the eigenvalues

    x = np.linspace(0.0011 , 10 , 5000)
    f = np.vectorize(lambda x : marchenko_pastur_pdf(x,lanbda,sigma=sigma))
    ax.plot(x,f(x), linewidth=4, color = 'r')

    plt.show()