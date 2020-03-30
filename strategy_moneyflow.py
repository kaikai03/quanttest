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
pd.set_option('display.float_format', lambda x: '%.3f' % x)


codes = ['002415']
start_date = '2018-01-15'
end_date = '2018-01-21'
# init_cash = 1000000
data = QA.QA_fetch_stock_day_adv(codes, start_date, end_date).to_hfq()
data = QA.QA_fetch_stock_min_adv(codes, start_date, end_date, frequence='5min').to_hfq()

a = data.amount * np.sign(data.close - data.close.shift())
a.groupby(["datetime",'code']).apply(np.sum)

data['Time'].groupby (data['Time'].map(lambda x: x[11:13])).count()

bar_size = datetime.timedelta(seconds=60*30)
time_group = pd.Grouper(level=0, freq=pd.Timedelta(bar_size))
a.groupby([time_group,"code"]).apply(np.sum)

bar_size = datetime.timedelta(seconds=60*5)
time_group = pd.Grouper(level=0, freq=pd.Timedelta(bar_size))
a.groupby([time_group,"code"]).apply(np.sum)

bar_size = datetime.timedelta(seconds=60*60*24)
time_group = pd.Grouper(level=0, freq=pd.Timedelta(bar_size))
a.groupby([time_group,"code"]).apply(np.sum)

#成交量/流通总量 = 换手率
# 资金流动强度  sosff = (p_i - p)/p * Q_i/Q = (AP - p)/p * TR
# p_i现价 p参照价  q_i成交量,q周期内总流通  AP周期内均价   TR换手率
# 价格可考虑加权移动平均，消除滞后性误差



# account = QA.QA_Account(QA.QA_util_random_with_topic("admin_"),QA.QA_util_random_with_topic("admin_co_"))
# broker = QA.QA_BacktestBroker()
# account.reset_assets(init_cash)


