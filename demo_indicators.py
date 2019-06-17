import QUANTAXIS as QA
import pandas as pd
import numpy as np
a = np.random.rand(5)
b = np.random.rand(5)
cov = np.cov(a,b)

DataFrame = Series = None
N= N1= N2=N3 = N4= N5=N6=M=M1 = 0
COND = 0
A,B = V1,V2 = 0,0
QA.MA(Series, N)
QA.EMA(Series, N)
QA.SMA(Series, N, M=1)
QA.DIFF(Series, N=1)
QA.HHV(Series, N)
QA.LLV(Series, N)
QA.SUM(Series, N)
QA.ABS(Series)
QA.MAX(A, B)
QA.MIN(A, B)
QA.CROSS(A, B)
QA.COUNT(COND, N)
QA.IF(COND, V1, V2)
QA.REF(Series, N)
QA.STD(Series, N)
QA.AVEDEV(Series, N)
QA.BBIBOLL(Series, N1, N2, N3, N4, N, M)

QA.QA_indicator_OSC(DataFrame, N, M)
QA.QA_indicator_BBI(DataFrame, N1, N2, N3, N4)
QA.QA_indicator_PBX(DataFrame, N1, N2, N3, N4, N5, N6)
QA.QA_indicator_BOLL(DataFrame, N)
QA.QA_indicator_ROC(DataFrame, N, M)
QA.QA_indicator_MTM(DataFrame, N, M)
QA.QA_indicator_KDJ(DataFrame, N=9, M1=3, M2=3)
QA.QA_indicator_MFI(DataFrame, N)
QA.QA_indicator_ATR(DataFrame, N)
QA.QA_indicator_SKDJ(DataFrame, N, M)
QA.QA_indicator_WR(DataFrame, N, N1)
QA.QA_indicator_BIAS(DataFrame, N1, N2, N3)
QA.QA_indicator_RSI(DataFrame, N1, N2, N3)
QA.QA_indicator_ADTM(DataFrame, N, M)
QA.QA_indicator_DDI(DataFrame, N, N1, M, M1)
QA.QA_indicator_CCI(DataFrame, N=14)


def JLHB(data, m=7, n=5):
    """
    通达信定义
    VAR1:=(CLOSE-LLV(LOW,60))/(HHV(HIGH,60)-LLV(LOW,60))*80;
    B:SMA(VAR1,N,1);
    VAR2:SMA(B,M,1);
    绝路航标:IF(CROSS(B,VAR2) AND B<40,50,0);
    """
    var1 = (data['close'] - QA.LLV(data['low'], 60)) / \
        (QA.HHV(data['high'], 60) - QA.LLV(data['low'], 60)) * 80
    B = QA.SMA(var1, m)
    var2 = QA.SMA(B, n)
    return pd.DataFrame({'JLHB':QA.CROSS(B,var2)*(B<40)})


QA.QA_fetch_stock_day_adv('000001','2017-01-01','2017-05-31').to_qfq().add_func(JLHB)

data = pd.DataFrame()
ind=data.add_func(QA.QA_indicator_WR,1,2)
inc=QA.QA_DataStruct_Indicators(ind)
inc.get_timerange('2018-01-07','2018-01-12','000001')
inc.get_code('000002')
inc.get_indicator('2018-01-12','000001','WR1') #某个指标值
inc.get_indicator('2018-01-12','000001') #所有指标值


def MACD_JCSC(dataframe,SHORT=12,LONG=26,M=9):
    """
    1.DIF向上突破DEA，买入信号参考。

    2.DIF向下跌破DEA，卖出信号参考。
    """
    CLOSE=dataframe.close
    DIFF =QA.EMA(CLOSE,SHORT) - QA.EMA(CLOSE,LONG)
    DEA = QA.EMA(DIFF,M)
    MACD =2*(DIFF-DEA)

    CROSS_JC=QA.CROSS(DIFF,DEA)
    CROSS_SC=QA.CROSS(DEA,DIFF)
    ZERO=0
    return pd.DataFrame({'DIFF':DIFF,'DEA':DEA,'MACD':MACD,'CROSS_JC':CROSS_JC,'CROSS_SC':CROSS_SC,'ZERO':ZERO})

ind=data.add_func(MACD_JCSC)

ind.xs('000001',level=1)['2016'].plot()