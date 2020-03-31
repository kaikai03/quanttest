# -*- coding: utf-8 -*-
import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QAParameter import MARKET_TYPE
import pandas as pd
import numpy as np

import datetime

import matplotlib.pyplot as plt


import backtest_base as trade

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


codes = ['002415','002416']
start_date = '2018-01-15'
end_date = '2018-01-21'
# init_cash = 1000000
data = QA.QA_fetch_stock_day_adv(codes, start_date, end_date).to_hfq()
data = QA.QA_fetch_stock_min_adv(codes, start_date, end_date, frequence='5min').to_hfq()

res_adv = QA.QA_fetch_financial_report_adv(codes,'2019-09-30',ltype='EN')
res_adv.data.stack()
res_adv.get_key('002415','2019-09-30','freeCirculationStock')
#国内一般只对可流通部分的股票计算换手率，一般在1~2.5，3为分界，表示活跃
listed_shares = res_adv.get_key(codes,'2019-09-30','listedAShares')
res_adv.get_key('002415','2019-09-30','totalCapital')



data.data
ret_rate = data.close_pct_change()
direction = np.sign(ret_rate)

bar_size = datetime.timedelta(seconds=60*60*24)
time_group = pd.Grouper(level=0, freq=pd.Timedelta(bar_size))
moneyflow = (data.amount * direction).groupby([time_group,"code"]).apply(np.sum)

exchange_rate = data.volume.groupby([time_group,"code"]).apply(lambda x: np.sum(x)/listed_shares[:,x.index[0][1]][0])
sosff = ret_rate.groupby([time_group,"code"]).apply(lambda x: np.sum(x)*exchange_rate[:,x.index[0][1]][0])  * 100

ic = abs(moneyflow)/data.amount.groupby([time_group,"code"]).apply(np.sum)

mfp = moneyflow/ (data.price.groupby([time_group,"code"]).apply(lambda x: np.average(x)*listed_shares[:,x.index[0][1]][0]))

mfl = (data.price.groupby([time_group,"code"]).apply(lambda x: np.average(x)*listed_shares[:,x.index[0][1]][0])).diff(len(data.index.levels[1])) / moneyflow

final = pd.DataFrame({'MF':moneyflow, 'RET':ret_rate.groupby([time_group,"code"]).apply(np.sum), "ExcR":exchange_rate, "SOSFF":sosff, "MFP":mfp, "MFL":mfl,"IC":ic})
final["MF"] = (final["MF"]-final["MF"].mean())/final["MF"].std()
final["MFL"] = (final["MFL"]-final["MFL"].mean())/final["MFL"].std()
final.plot()
final["RET"].plot()
final["MF"].plot()


#成交量/流通总量 = 换手率
# 资金流动强度  sosff = (p_i - p)/p * Q_i/Q = (AP - p)/p * TR
# p_i现价 p参照价  q_i成交量,q周期内总流通  AP周期内均价   TR换手率
# 价格可考虑加权移动平均，消除滞后性误差

#ic = abs(mf)/amount
#资金流强度mfp=mf/value 流通市值

#资金流杠杆倍数 mfl= delta value / mf
#r



# account = QA.QA_Account(QA.QA_util_random_with_topic("admin_"),QA.QA_util_random_with_topic("admin_co_"))
# broker = QA.QA_BacktestBroker()
# account.reset_assets(init_cash)

