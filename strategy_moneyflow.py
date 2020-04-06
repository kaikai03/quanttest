# -*- coding: utf-8 -*-
import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QAParameter import MARKET_TYPE
import pandas as pd
import numpy as np

import datetime

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

import matplotlib.pyplot as plt

import backtest_base as trade

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

codes = ['002415']
codes = ['002415', '002416', '002417']

start_date = '2018-01-15'
end_date = '2018-06-21'
# init_cash = 1000000
data = QA.QA_fetch_stock_day_adv(codes, start_date, end_date).to_hfq()
data = QA.QA_fetch_stock_min_adv(codes, start_date, end_date, frequence='5min').to_hfq()
data.data
data.index

def get_company_name(codes):
    infos = QA.QA_fetch_stock_list_adv()
    return [(code, infos[infos.code==code].name[0]) for code in codes]

QA.QA_fetch_stock_block_adv().block_name
codes = QA.QA_fetch_stock_block_adv().get_block('上证50').code
get_company_name(codes)


def MF(data_, hour=1):
    codes_ = list(data_.index.levels[1])
    res_adv = QA.QA_fetch_financial_report_adv(codes_, '2019-09-30', ltype='EN')
    # 国内一般只对可流通部分的股票计算换手率，一般在1~2.5，3为分界，表示活跃
    listed_shares = res_adv.get_key(codes_, '2019-09-30', 'listedAShares')
    # freeCirculationStock, totalCapital

    # print("ffffffffffffffffffffffffffff",codes_)

    ret_rate = (data_.close - data_.preclose)/data_.preclose
    direction = np.sign(ret_rate)

    bar_size = datetime.timedelta(seconds=60 * 60 * hour)
    time_group = pd.Grouper(level=0, freq=pd.Timedelta(bar_size))
    moneyflow = (data_.amount * direction).groupby([time_group, "code"]).apply(np.sum)

    moneyflow_in = (data_.amount * direction).groupby([time_group, "code"]).apply(lambda x: np.sum(x[x > 0]))
    moneyflow_out = (data_.amount * direction).groupby([time_group, "code"]).apply(lambda x: np.sum(x[x < 0]))

    speed = data_.amount.groupby([time_group, "code"])\
        .apply(lambda x: np.sum(np.diff(np.divide(np.subtract(x[1:], x[0:-1]),x[0:-1]))))


    exchange_rate = data_.volume.groupby([time_group, "code"]) \
        .apply(lambda x: np.sum(x) / listed_shares[:, x.index[0][1]][0])

    sosff = ret_rate.groupby([time_group, "code"]) \
        .apply(lambda x: np.sum(x) * exchange_rate[:, x.index[0][1]][0]) * 100

    ic = abs(moneyflow) / data_.amount.groupby([time_group, "code"]).apply(np.sum)

    mfp = moneyflow / \
        (data_.close.groupby([time_group, "code"]).apply(lambda x: np.average(x) * listed_shares[:, x.index[0][1]][0]))

    mfl = (data_.close.groupby([time_group, "code"])
           .apply(lambda x: np.average(x) * listed_shares[:, x.index[0][1]][0])) \
        .diff(1) / moneyflow

    return pd.DataFrame({'MF': moneyflow,
                         'MF_IN': moneyflow_in,
                         'MF_OUT': moneyflow_out,
                         'SPEED': speed,
                         'CLOSE': data_.close.groupby([time_group, "code"]).apply(lambda x: x[-1]),
                         'RET': ret_rate.groupby([time_group, "code"]).apply(np.sum),
                         "ExcR": exchange_rate, "SOSFF": sosff, "MFP": mfp, "MFL": mfl, "IC": ic})


ind = data.add_func(MF, 4).pivot_table(index=['datetime', 'code'])
final = ind

final["MF"] = (final["MF"] - final["MF"].mean()) / final["MF"].std()
final["MFL"] = (final["MFL"] - final["MFL"].mean()) / final["MFL"].std()
final.plot()
final["RET"].plot()
final["MF"].plot()

data.groupby(level=1, sort=False).apply(MF, 24)

final.loc[pd.IndexSlice[:, '002417'],:]


np.subtract(final.loc[pd.IndexSlice[:, '601012'],"MFP"][1::2], final.loc[pd.IndexSlice[:, '601012'],"MFP"][::2])

for code in final.index.levels[1]:
    print(code, np.corrcoef(np.subtract(final.loc[pd.IndexSlice[:, code],"MFP"][1::2]*0.1, final.loc[pd.IndexSlice[:, code],"MFP"][::2]),
                            final.loc[pd.IndexSlice[:, code], "RET"][1::2])[0][1])


final.loc[pd.IndexSlice[:, '002415'], ["MFP","RET"]]\
    .groupby([pd.Grouper(level=0, freq=pd.Timedelta(datetime.timedelta(seconds=60 * 60 * 2))),'code'])\
    .apply(np.sum)

speed = final.loc[pd.IndexSlice[:, '002415'],"SPEED"][0::2]
cur_ret = final.loc[pd.IndexSlice[:, '002415'],"RET"][0::2]
mfp = final.loc[pd.IndexSlice[:, '002415'],"MFP"][0::2]

ret = final.loc[pd.IndexSlice[:, '002415'],"RET"][1::2]

ret[ret>=0]=1
ret[ret<0]=0
ret = list(ret)
x = [[x, cur_ret[i], mfp[i]] for i, x in enumerate(speed)]

reg = linear_model.LogisticRegression()
reg.fit(x, ret)
reg.score(x, ret)
reg.predict(x)

data.close.groupby([pd.Grouper(level=0, freq=pd.Timedelta(datetime.timedelta(seconds=60 * 60 * 0.5))),'code'])\
    .apply(lambda x: x[-2])


final.pivot_table(index=['code','datetime'])

# 成交量/流通总量 = 换手率
# 资金流动强度  sosff = (p_i - p)/p * Q_i/Q = (AP - p)/p * TR
# p_i现价 p参照价  q_i成交量,q周期内总流通  AP周期内均价   TR换手率
# 价格可考虑加权移动平均，消除滞后性误差

# ic = abs(mf)/amount
# 资金流强度mfp=mf/value 流通市值

# 资金流杠杆倍数 mfl= delta value / mf
# r


# account = QA.QA_Account(QA.QA_util_random_with_topic("admin_"),QA.QA_util_random_with_topic("admin_co_"))
# broker = QA.QA_BacktestBroker()
# account.reset_assets(init_cash)
