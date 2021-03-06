﻿# coding=utf-8

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
# # result = pd.merge(stock_diff,infos.loc[codes]['name'], left_index=True,right_index=True)
#
# QA.QA_fetch_stock_block_adv().get_block('雄安新区').data
#
# codes = QA.QA_fetch_stock_block_adv().get_block(['生物医药','化学制药']).code
# QA.QA_fetch_stock_block_adv().data
# data =QA.QA_fetch_stock_day_adv(codes[1:10],'2017-01-05','2017-12-25').to_hfq()
#
#
# data =QA.QA_fetch_stock_day_adv(['002821','300340','300452','300482'],'2017-01-05','2017-12-25').to_hfq()


####demo#####

# lgR_mat,samples = preprocess(data)
#
# m = marchenko_pastur_optimize(lgR_mat,samples)
# m.fit()
# m.fit(only_keep_max_side=False)
# m.sigma
# m.lanbda
# m.filt_min_var_weights
# m.normal_min_var_weights
# m.filt_min_var_weights_series.sum()
# m.normal_min_var_weights_series.sum()
#
# m.filt_min_var_weights_series_norm.sum()
# m.normal_min_var_weights_series_norm.sum()
#
#
#
#
# m.show_marchenko_pdf_plot()
# m.show_filtered_compare_plot()
# m.show_weights_compare_plot()
# m.show_all_plot()
####demo#####

def preprocess(data):
    compare_samples = data.close.unstack()
    lost_too_much = compare_samples.columns[compare_samples.isnull().sum() > compare_samples.shape[0]*0.1]
    compare_samples.drop(lost_too_much, axis=1,inplace=True)
    compare_samples.dropna(0, inplace= True)

    print("lost too must,del:{}".format(lost_too_much))

    lgR = np.log((data.close/data.pre_close).unstack())
    lgR.drop(lost_too_much, axis=1,inplace=True)
    lgR.dropna(0, inplace= True)
    lgR_mat = lgR
    return lgR,compare_samples

class marchenko_pastur_optimize:
    """输入数据矩阵，返回风险最低时对应的权重"""
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
        N,T = codes_dates_df.shape
        self.lanbda = T/N

        self.corMat = None

        self.eigVals = None
        self.eigVects = None

        self.fitted = False
        self.filtered_matrix = None
        self.filtered_cov = None

        self.filt_min_var_weights = None
        self.normal_min_var_weights = None

        self._filt_min_var_weights_series = None
        self._normal_min_var_weights_series = None

        self._filt_min_var_weights_series_norm = None
        self._normal_min_var_weights_series_norm = None


    @property  ##获取过滤后的权重Series的懒加载
    def filt_min_var_weights_series(self):
        if not isinstance(self._filt_min_var_weights_series, type(None)):
            return self._filt_min_var_weights_series
        else:
            self._filt_min_var_weights_series = pd.Series(self.filt_min_var_weights, self.data.columns.values.tolist())
            return self._filt_min_var_weights_series

    @property  ##获取原始最小权重Series的懒加载
    def normal_min_var_weights_series(self):
        if not isinstance(self._normal_min_var_weights_series, type(None)):
            return self._normal_min_var_weights_series
        else:
            self._normal_min_var_weights_series = pd.Series(self.normal_min_var_weights, self.data.columns.values.tolist())
            return self._normal_min_var_weights_series

    @property  ##获取归一化的滤后权重
    def filt_min_var_weights_series_norm(self):
        if not isinstance(self._filt_min_var_weights_series_norm, type(None)):
            return self._filt_min_var_weights_series_norm
        else:
            weights=self.filt_min_var_weights_series.copy()
            weights[weights<=0] = np.nan
            weights.dropna(0, inplace= True)
            weights=weights / weights.sum()
            self._filt_min_var_weights_series_norm = weights

            return self._filt_min_var_weights_series_norm

    @property  ##获取归一化的滤前权重
    def normal_min_var_weights_series_norm(self):
        if not isinstance(self._normal_min_var_weights_series_norm, type(None)):
            return self._normal_min_var_weights_series_norm
        else:
            weights=self.normal_min_var_weights_series.copy()
            weights[weights<=0] = np.nan
            weights.dropna(0, inplace= True)
            weights=weights / weights.sum()
            self._normal_min_var_weights_series_norm = weights

            return self._normal_min_var_weights_series_norm


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
