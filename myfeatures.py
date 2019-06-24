# coding=utf-8

import matplotlib
import QUANTAXIS as QA
import pandas as pd
import numpy as np

##### base
# QA.MA(Series, N)
# QA.EMA(Series, N)
# QA.SMA(Series, N, M=1)
# QA.DIFF(Series, N=1)
# QA.HHV(Series, N)
# QA.LLV(Series, N)
# QA.SUM(Series, N)
# QA.ABS(Series)
# QA.MAX(A, B)
# QA.MIN(A, B)
# QA.CROSS(A, B)
# QA.COUNT(COND, N)
# QA.IF(COND, V1, V2)
# QA.REF(Series, N)
# QA.STD(Series, N)
# QA.AVEDEV(Series, N)
# QA.BBIBOLL(Series, N1, N2, N3, N4, N, M)
# QA.QA_indicator_OSC(DataFrame, N, M)
# QA.QA_indicator_BBI(DataFrame, N1, N2, N3, N4)
# QA.QA_indicator_PBX(DataFrame, N1, N2, N3, N4, N5, N6)
# QA.QA_indicator_BOLL(DataFrame, N)
# QA.QA_indicator_ROC(DataFrame, N, M)
# QA.QA_indicator_MTM(DataFrame, N, M)
# QA.QA_indicator_KDJ(DataFrame, N=9, M1=3, M2=3)
# QA.QA_indicator_MFI(DataFrame, N)
# QA.QA_indicator_ATR(DataFrame, N)
# QA.QA_indicator_SKDJ(DataFrame, N, M)
# QA.QA_indicator_WR(DataFrame, N, N1)
# QA.QA_indicator_BIAS(DataFrame, N1, N2, N3)
# QA.QA_indicator_RSI(DataFrame, N1, N2, N3)
# QA.QA_indicator_ADTM(DataFrame, N, M)
# QA.QA_indicator_DDI(DataFrame, N, N1, M, M1)
# QA.QA_indicator_CCI(DataFrame, N=14)
####################

def JLHB(data, m=7, n=5):
    """
    绝路航标-通达信定义
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

def MACD_JCSC(dataframe, SHORT=12, LONG=26, M=9):
    """
    金叉死叉
    1.DIF向上突破DEA，买入信号参考。
    2.DIF向下跌破DEA，卖出信号参考。
    """
    CLOSE = dataframe.close
    DIFF = QA.EMA(CLOSE, SHORT) - QA.EMA(CLOSE, LONG)
    DEA = QA.EMA(DIFF, M)
    MACD = 2*(DIFF-DEA)

    CROSS_JC = QA.CROSS(DIFF, DEA)
    CROSS_SC = QA.CROSS(DEA, DIFF)
    ZERO = 0
    return pd.DataFrame({'DIFF': DIFF, 'DEA': DEA, 'MACD': MACD, 'CROSS_JC': CROSS_JC, 'CROSS_SC': CROSS_SC, 'ZERO': ZERO})

def my(data, m=7, n=5):
    var1 = data['close'].diff(1)/data['close'].shift(1) *100
    var2 = data['close'].diff(2)/data['close'].shift(2) *100
    B = QA.SMA(var1, m)
    B2 = QA.SMA(var2, m)
    return pd.DataFrame({'diff1':var1, 'diff2':var2, 'B':B, 'B2':B2 })

if __name__ == '__main__':
    ind=QA.QA_fetch_stock_day_adv('000001','2017-01-01','2017-05-31').to_qfq().add_func(JLHB,7,5)
    inc=QA.QA_DataStruct_Indicators(ind)
    inc.get_indicator('2017-05-25','002415')
    inc.get_timerange('2018-01-07','2018-01-12','000001')
    inc.get_code('002415')
    inc.get_indicator('2017-05-25','002415','B')

    data = QA.QA_fetch_stock_day_adv('002415','2017-01-01','2017-05-31').to_qfq()
    ind=QA.QA_fetch_stock_day_adv('002415','2017-01-01','2017-05-31').to_qfq().add_func(my,7,5)
    inc.get_timerange('2017-05-25','2017-05-26','002415')

    pass
