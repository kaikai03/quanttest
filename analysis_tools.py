from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
from scipy.optimize import leastsq
import pandas as pd


import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic  import acorr_ljungbox
from arch import arch_model

##dataBase                 ###########
s0001 = {}
days = {}


### GARCH               ###########
def GARCHTest():
    s0001 = days[days["code"]=="000001"]
    time_serie = s0001['close'].reset_index(level='code',drop=True)
    time_serie=pd.Series([10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
    6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
    10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
    12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
    13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
    9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
    11999,9390,13481,14795,15845,15271,14686,11054,10395])

    drawplot_easy(time_serie)
    ## 对数变换主要是为了减小数据的振动幅度，使其线性规律更加明显
    # 对数变换相当于增加了一个惩罚机制，数据越大其惩罚越大，数据越小惩罚越小。
    time_serie_lg = np.log(time_serie)
    drawplot_easy(time_serie_lg)

    time_serie = pd.Series(time_serie).diff(1).shift(-1).dropna()
    time_serie_lg_diff = time_serie_lg.diff(1).dropna()
    drawplot_easy(time_serie_lg_diff)

    ## 移动平均
    rol_mean = time_serie_lg.rolling(window=10).mean()
    drawplot_easy(rol_mean)
    ## 指数加权移动平均
    # 加权系数以指数式递减的，即各指数随着时间而指数式递减。
    rol_weighted_mean = pd.ewma(time_serie_lg, span=10)
    drawplot_easy(rol_weighted_mean)


    ##趋势、周期性处理
    # seasonal_decompose 的index必须是DatetimeIndex,否则报错

    # decomposition =smt.seasonal_decompose(time_serie_lg,freq=30)
    #
    # trend = decomposition.trend
    # seasonal = decomposition.seasonal
    # residual = decomposition.resid
    #
    # drawplot_easy([trend,seasonal,residual], labels=["trend","seasonal","residual"])
    # drawplot_easy(residual)
    # drawplot_easy(seasonal)
    # drawplot_easy(trend)


    ## the bigger the p the more reason we assert that there is a unit root
    # 单位根既非平稳
    ADFresult = smt.adfuller(time_serie_lg_diff)
    output = pd.Series(ADFresult[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in ADFresult[4].items():
        output['Critical Value (%s)'%key] = value
    output

    ## p<0.05, 序列为非白噪声 (lags参数:一阶差分后)
    acorr_ljungbox(result.resid,lags=30)

    ##  pacf，acf 定阶
    ar_p = smt.pacf(time_serie,nlags=10)
    ma_q = smt.acf(time_serie,nlags=10,qstat=True)


    fig = plt.figure(2,figsize=(9,6))
    fig.subplots_adjust(hspace=0.8)
    origin_ax = fig.add_subplot(311)
    time_serie_lg_diff.plot(ax=origin_ax,title='time series')
    pacf_ax = fig.add_subplot(323)
    plot_pacf(time_serie_lg_diff,lags=10,ax=pacf_ax)
    acf_ax = fig.add_subplot(324)
    plot_acf(time_serie_lg_diff,lags=10,ax=acf_ax)
    qq_ax = fig.add_subplot(325)
    qq_ax.set_title('QQ Plot')
    sm.qqplot(time_serie_lg_diff,ax=qq_ax)
    pp_ax = fig.add_subplot(326)
    scs.probplot(time_serie_lg_diff,
                 sparams=(time_serie_lg_diff.mean(), time_serie_lg_diff.std()),
                 plot=pp_ax)

    ## 生成序列
    # a = smt.arma_generate_sample(ar_p, ma_q[0], s0001['close'].size)
    # drawplot_easy([a.tolist()],labels=['a'])

    a+=4
    # p, d, q = order
    ##如果用来预测的话可用log_diff 找p q。之后ARMA用原始数据
    ##但不稳定性太强不适合做预测
    model = smt.ARIMA(time_serie_lg_diff,order=(1, 0, 0))

    result = model.fit(disp=-1)

    ## 结果
    result.summary2()
    result.forecast(5)
    result.params
    result.aic, result.bic, result.hqic
    result.arparams, result.arroots, result.maparams
    result.bse, result.pvalues
    result.resid

    orign = time_serie_lg_diff.reset_index(drop = True)
    fitted = result.fittedvalues

    drawplot_easy(fitted, labels="fittedvalues")
    drawplot_easy(pd.Series(fitted).shift(-1), labels="fittedvalues-1")
    drawplot_easy(orign, labels="orign")
    drawplot_easy(result.predict(0,486+5), labels="predict")
    result.k_exog
    result.k_trend
    (result.resid**2).sum()

    drawplot_easy(np.exp(fitted+time_serie_lg.shift(-1).dropna()), labels="recover")

### plot             ###########
def drawplot_df(df, groupby='code', price='close',x_tick='date',  split=False):
    '''
    以groupby为单位，绘制dataframe的matplot图，
    price：选择用于绘图的列名，默认为收盘价
    x_tick: 选择绘制x轴的数据，默认用日期
    split：各个code分成各子图绘制，不全绘制在一个图中
    '''
    count = len(df.groupby(level = "code").size().index)
    fig = plt.figure(count,figsize=(8,6))
    fig.subplots_adjust(hspace=0.5)
    if split:
        ## 如果分别显示，将以方阵显示各个子图
        square_matrix_unit = np.ceil(np.sqrt(count))
        param_subplot = square_matrix_unit*100 + square_matrix_unit*10 +1
    else:
        param_subplot = 111

    for i,data_in_a_code in enumerate(df.groupby(level = groupby)):
        color_=None  ## 随机色np.random.rand(3,)
        label_ = data_in_a_code[0]

        if split:
            ## 分别设置各个子图状态
            ax=plt.subplot(param_subplot+i)
            ax.xaxis.grid(True, which='minor')
            ax.yaxis.grid(True, which='major')
            ax.set_title(label_,fontsize='small')
             ## 小图不显示x轴标签，金融场景下一般是时间轴，
            ## 显示效果实在........
            ax.set_xticklabels([],
                               horizontalalignment='left',
                               fontsize='small')
            plt.yticks(fontsize='small')
        else:
            ## 不分别绘制时，在循环外统一设置
            if i ==0:
                ax=plt.subplot(param_subplot)

        # plt.plot(data_in_a_code[1][x_tick], data_in_a_code[1][price],
        #          label=label_, color=color_,linewidth=0.5)
        plt.plot(data_in_a_code[1].index.get_level_values(x_tick), data_in_a_code[1][price],
                 label=label_, color=color_,linewidth=0.5)

    if not split:
        ax.xaxis.grid(True, which='minor')
        ax.yaxis.grid(True, which='major')
        plt.legend(loc="upper right", frameon=False,fancybox=False)
        plt.xticks(rotation=20)

    plt.show()

def drawplot_xy(xs, ys,labels=None,title=None):
    fig = plt.figure(1)
    ax=plt.subplot(111)
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='major')
    if title != None:
        ax.set_title(title)

    ##为二维数组时，判定为多条线段绘制
    if isinstance(xs[0], list):
        for i in range(len(xs)):
            label_ = None
            if isinstance(labels, list):
                label_ = labels[i]
            plt.plot(xs[i], ys[i],label=label_,linewidth=0.5)
    else:
        plt.plot(xs, ys, label=labels, linewidth=0.5)

    plt.legend(loc="upper right", frameon=False,fancybox=False)
    plt.show()

def drawplot_easy(xs,labels=None,title=None):
    fig = plt.figure(1)
    ax=plt.subplot(111)
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='major')
    if title != None:
        ax.set_title(title)

    ##为二维数组时，判定为多条线段绘制
    if isinstance(xs[0], list) or isinstance(xs[0], pd.core.series.Series):
        for i in range(len(xs)):
            label_ = None
            if isinstance(labels, list):
                label_ = labels[i]
            plt.plot(xs[i], label=label_,linewidth=0.5)
    else:
        plt.plot(xs, label=labels, linewidth=0.5)

    plt.legend(loc="best", frameon=False,fancybox=False)
    plt.show()

### hurst              ###########
def hurst(ts):
    """标准差取对数方法的实现
    >0.5 ,记忆强, 未来的增量和过去的增量相关,继续保持现有趋势的可能性强
    < 0.5 ,很有可能是记忆的转弱 ,趋势结束和反转的开始(mean reversion)
    数值越靠近0.5说明随机性越强 ,无法判定走向(Random Walk)
    """
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    drawplot_xy(np.log(lags), np.log(tau))
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def avg_allocation_data(df, block_count, flag_label='flag'):
    '''加入一组flag(列)，将数据均分成N组
    flag in [1,2,3...]
    -1为无法均分多余数据，会出现在数据最开始部分。
    :block_count  分块的数量
    :colume_label  flag列的列名,
        如果原数据内有同名列，将直接覆盖，
        否则新增
    :return  每个子块的长度
    '''
    if block_count < 1:
        df[flag_label] = -1
        return

    if block_count == 1:
        df[flag_label] = 1
        return

    df[flag_label] = -1
    count = df.shape[0]
    if block_count > count:
        raise 'avg_allocation_data, block_count large than data size'
    N = int(count/block_count)
    redundancy = count - N*block_count
    for i in range(block_count):
        df[N*i+redundancy : N*(i+1)+redundancy][flag_label] = i+1
    return N

def RS_simple(df, Ns):
    '''重标极差(R/S)采样观察值
    :Ns 分块块数数组[n1,n2...]
    :return [(n1,RS1),(n2,RS2),...]
        n分块数，RS是分块数对应的重标极差
    '''

    RS_resualts = []
    for N in Ns:
        # 数据分块，并记录每子块长度
        n = avg_allocation_data(df,N,flag_label='flag')
        print("RS_simple block count:",N,"size:",n)
        R_list = np.zeros(N)  #记录每组的极差
        S_list = np.zeros(N)  #记录每组的方差
        for i in range(N):
            ##求 每组的离差累和、极差R、标准差S
            close_region = df[df['flag']==i+1]["close"]
            licha = close_region - close_region.mean()
            licha_leihe = licha.cumsum()
            R = licha_leihe.max()-licha_leihe.min()
            R_list[i] = R
            S_list[i] = close_region.std()

        RS = (R_list/S_list).mean()
        print(R_list)
        print(S_list)
        print(n)
        print(RS)
        RS_resualts.append((n,RS))
    return RS_resualts

def hurst_RS(df):
    '''基于重标极差（R/S）分析方法的Hurst指数
    '''
    ##重标极差（R/S）采样观察值
    simples = zip(*RS_simple(df,[2,4,8,16,24]))
    simples_list = list(simples)
    # drawplot_xy(np.log(simples_list[0]), np.log(simples_list[1]),title='origin price')
    drawplot_xy(simples_list[0], simples_list[1],title='compounded return rate')

    ## 最小二乘估计常数K,和Hurst指数
    #RS = Kn^H
    #log(RS) = log(K)+H*log(n)
    def error(p, n, RS):
        def hurst(p,n):
            K, H = p
            return np.log(K) + H*np.log(n)
        return hurst(p, n) - np.log(RS)

    p0 = [10,0.5]
    Para = leastsq(error,p0,args=(simples_list[0],simples_list[1]))
    K, H = Para[0]
    return H


