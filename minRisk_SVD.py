# coding=utf-8

# https://srome.github.io/Eigenvesting-III-Random-Matrix-Filtering-In-Finance/
# 不过文章里分布函数写错了。
# 最终换成这篇的。http://www.doc88.com/p-9877831718045.html

import matplotlib
import QUANTAXIS as QA
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
# np.linalg.svd()
import sklearn.decomposition as skd
import matplotlib.pyplot as plt

# infos = QA.QA_fetch_stock_list_adv()
# result = pd.merge(stock_diff,infos.loc[codes]['name'], left_index=True,right_index=True)

codes = QA.QA_fetch_stock_block_adv().get_block(['生物医药','化学制药']).code
data =QA.QA_fetch_stock_day_adv(codes[1:80],'2017-01-05','2017-12-25').to_hfq()


# data =QA.QA_fetch_stock_day_adv(['002415','601155','000735','300558'],'2017-01-05','2017-12-25').to_hfq()

# data.price_chg.values
# data.price_chg[:, '000735'].values
# data.price_chg['2017-01-10']
# data.data.loc["2017-01-03","000735"]
#
# data.price_chg.unstack().values.T

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


samples = data.close.unstack()
lost_too_much = samples.columns[samples.isnull().sum() > samples.shape[0]*0.1]
samples.drop(lost_too_much, axis=1,inplace=True)
samples.dropna(0, inplace= True)

lgR = np.log((data.close/data.pre_close).unstack())
lgR.drop(lost_too_much, axis=1,inplace=True)
lgR.dropna(0, inplace= True)
lgR_mat = lgR


m = marchenko_pastur_optimize(lgR_mat,samples)
m.fit()
m.sigma
m.lanbda
m.filt_min_var_weights
m.normal_min_var_weights
m.show_marchenko_pdf_plot()
m.show_filtered_compare_plot()
m.show_weights_compare_plot()
m.show_all_plot()

class marchenko_pastur_optimize:
    def __init__(self, codes_dates_df, compare_df=None):
        if codes_dates_df.isnull().any().sum() != 0:
            raise Exception("codes_dates_df contant NaN")
        if not isinstance(compare_df, type(None)):
            if compare_df.isnull().any().sum() != 0:
                raise Exception("compare_df contant NaN")

        self.data = codes_dates_df
        self.origin_data_compare = compare_df
        self.sigma = 1
        self.only_keep_max_side=True
        N,T = lgR_mat.shape
        self.lanbda = T/N

        self.corMat = None

        self.eigVals = None
        self.eigVects = None

        self.fitted = False
        self.filtered_matrix = None
        self.filtered_cov = None

        self.filt_min_var_weights = None
        self.normal_min_var_weights = None

    # 马尔琴科分布理论上界
    def marchenko_pastur_maxmum(self, lanbda, sigma=1):
        return np.power(sigma*sigma*(1 + np.sqrt(lanbda)),2)
    # 马尔琴科分布理论下界
    def marchenko_pastur_minmum(self, lanbda, sigma=1):
        return np.power(sigma*sigma*(1 - np.sqrt(lanbda)),2)
    # Definition of the Marchenko-Pastur density
    def marchenko_pastur_pdf(self, x, lanbda, sigma=1):
        b=self.marchenko_pastur_maxmum(lanbda, sigma) # Largest eigenvalue
        a=self.marchenko_pastur_minmum(lanbda, sigma)# Smallest eigenvalue
        return (1/(2*np.pi*sigma*sigma*lanbda*x))*np.sqrt((b-x)*(x-a))*(0 if (x > b or x <a ) else 1)


    def fit(self, sigma=1, only_keep_max_side=True):
        """only_keep_max_side 默认过滤掉马尔琴科分布内及其左边的数据。
        当only_keep_max_side=false时，马尔琴科分布两侧的信号将被保留。
        """
        self.sigma = sigma
        self.only_keep_max_side=only_keep_max_side
        self.corMat = np.mat(self.data.interpolate().corr())
        # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        self.eigVals, self.eigVects=np.linalg.eig(self.corMat)

        # 求马尔琴科分布理论边界
        max_theoretical_eval = self.marchenko_pastur_maxmum(self.lanbda, self.sigma)

        # Filter the eigenvalues out
        eigVals_filter = self.eigVals.copy()
        if only_keep_max_side:
            eigVals_filter[eigVals_filter <= max_theoretical_eval] = 0
        else:
            min_theoretical_eval = self.marchenko_pastur_minmum(self.lanbda, self.sigma)
            eigVals_filter[(eigVals_filter <= max_theoretical_eval) & (eigVals_filter >= min_theoretical_eval)] = 0


        self.filtered_matrix = np.dot(self.eigVects, np.dot(np.diag(eigVals_filter), self.eigVects.T))
        np.fill_diagonal(self.filtered_matrix, 1)

        # 求标准差
        variances = np.diag(np.cov(self.data,rowvar=0))
        standard_deviations = np.sqrt(variances)

        ## 求新的协方差矩阵
        ##  ∑ 展开可以写成  ∑∑ ρ_ij * σ_i * σ_j
        ## 所以可以用上面算好的相关系数矩阵，求解新的协方差矩阵。
        ## ∑` = √(∑*I) Ρ √(∑*I)
        self.filtered_cov = np.dot(np.diag(standard_deviations),
        np.dot(self.filtered_matrix,np.diag(standard_deviations)))

        ## Construct minimum variance weights
        ##  ∑^(-1)*1   /  (∑^(-1) * 1 ) * 1
        ##  顺手做个np.asarray转换，防止后面matrix没法画图
        filt_inv_cov = np.linalg.pinv(np.asarray(self.filtered_cov))
        ones = np.ones(len(filt_inv_cov))
        inv_dot_ones = np.dot(filt_inv_cov, ones)
        self.filt_min_var_weights = inv_dot_ones/ np.dot( inv_dot_ones , ones)
        self.fitted = True

    @property
    def normal_min_var_weights(self):
        if isinstance(self.origin_data_compare, type(None)):
            raise  Exception("compare_df is None, the value need this param")
        if not isinstance(self._normal_min_var_weights, type(None)):
            return self._normal_min_var_weights
        else:
            covariance_matrix = np.cov(self.origin_data_compare, rowvar=0)
            inv_cov_mat = np.linalg.pinv(covariance_matrix)

            ## Construct minimum variance weights
            ##  min  w∑w  ;  s.t ∑w = 1 （第一个sigma是协方差，第二个是和）
            ## 最小化资产组合风险的凸优化问题，解得权重
            ##  ∑^(-1)*1  /  (∑^(-1) * 1 ) * 1
            ones = np.ones(len(inv_cov_mat))
            inv_dot_ones = np.dot(inv_cov_mat, ones)
            self._normal_min_var_weights = inv_dot_ones / np.dot( inv_dot_ones , ones)
            return self._normal_min_var_weights

    @normal_min_var_weights.setter
    def normal_min_var_weights(self, n):
        self._normal_min_var_weights = n

    def show_marchenko_pdf_plot(self):
        if self.fitted == False:
            raise Exception("fit first")
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_autoscale_on(True)
        ax.hist(self.eigVals, density=True, bins=20) # Histogram the eigenvalues

        x = np.linspace(0.0011 , 12 , 5000)
        f = np.vectorize(lambda x : self.marchenko_pastur_pdf(x,self.lanbda,sigma=self.sigma))
        ax.plot(x,f(x), linewidth=4, color = 'r')

        plt.show()

    # 显示过滤前相关性矩阵图和过滤后图
    def show_filtered_compare_plot(self):
        if self.fitted == False:
            raise Exception("fit first")

        f = plt.figure()
        ax = plt.subplot(121)
        ax.imshow(self.corMat)
        plt.title("Original")
        ax = plt.subplot(122)
        plt.title("Filtered")
        a = ax.imshow(self.filtered_matrix)
        # cbar = f.colorbar(a, ticks=[-1, 0, 1])
        plt.show()

    # 显示过滤前权重图和过滤后图
    def show_weights_compare_plot(self):
        if self.fitted == False:
            raise Exception("fit first")
        if isinstance(self.origin_data_compare, type(None)):
            raise  Exception("compare_df is None, the plot need this param")

        plt.figure(figsize=(14,6))
        ax = plt.subplot(121)
        min_var_portfolio = pd.DataFrame(data= self.normal_min_var_weights,
                                         columns = ['Investment Weight'],
                                         index = self.origin_data_compare.columns.values.tolist())
        min_var_portfolio.plot(kind = 'bar', ax = ax)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        plt.title('Minimum Variance')

        ax = plt.subplot(122)
        filt_min_var_portfolio = pd.DataFrame(data= self.filt_min_var_weights,
                                         columns = ['Investment Weight'],
                                         index = self.data.columns.values.tolist())
        filt_min_var_portfolio.plot(kind = 'bar', ax = ax)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        plt.title('Filtered Minimum Variance')
        plt.show()

    # 显示所有图
    def show_all_plot(self):
        if self.fitted == False:
            raise Exception("fit first")
        if isinstance(self.origin_data_compare, type(None)):
            raise  Exception("compare_df is None, the plot need this param")


        title = "σ:"+ str(self.sigma)\
                + "     λ:"+ str(self.lanbda)\
                + "     only_keep_max_side:" + str(self.only_keep_max_side)
        fig = plt.figure(title, figsize=(8,9))
        fig.set_tight_layout(True)

        ax  = fig.add_subplot(411)
        ax.set_autoscale_on(True)
        ax.hist(self.eigVals, density=True, bins=20) # Histogram the eigenvalues

        x = np.linspace(0.0011 , 12 , 5000)
        f = np.vectorize(lambda x : self.marchenko_pastur_pdf(x,self.lanbda,sigma=self.sigma))
        ax.plot(x,f(x), linewidth=4, color = 'r')

        ###################
        ax = plt.subplot(423)
        ax.imshow(self.corMat)
        plt.title("Original")
        ax = plt.subplot(424)
        plt.title("Filtered")
        a = ax.imshow(self.filtered_matrix)
        # cbar = f.colorbar(a, ticks=[-1, 0, 1])

        ax = plt.subplot(413)
        min_var_portfolio = pd.DataFrame(data= self.normal_min_var_weights,
                                         columns = ['Investment Weight'],
                                         index = self.origin_data_compare.columns.values.tolist())
        min_var_portfolio.plot(kind = 'bar', ax = ax)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.title('Minimum Variance')

        ax = plt.subplot(414)
        filt_min_var_portfolio = pd.DataFrame(data= self.filt_min_var_weights,
                                         columns = ['Investment Weight'],
                                         index = self.data.columns.values.tolist())
        filt_min_var_portfolio.plot(kind = 'bar', ax = ax)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
        plt.title('Filtered Minimum Variance')

        plt.subplots_adjust(hspace=0.4)
        plt.show()
