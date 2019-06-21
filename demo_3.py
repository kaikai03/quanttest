import QUANTAXIS as QA
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# 通过板块取代码
codes = QA.QA_fetch_stock_block_adv().get_block(['生物医药','化学制药']).code
for i in QA.QA_fetch_stock_block_adv().block_name:print(i)

# 取得股票信息
infos = QA.QA_fetch_stock_list_adv()

stock_date1 =QA.QA_fetch_stock_day_adv(codes, '2019-01-02',)
stock_date2 =QA.QA_fetch_stock_day_adv(codes, '2019-06-14')

stock_date =QA.QA_fetch_stock_day_adv(codes, '2019-01-02','2019-06-14').to_hfq()

# 计算总回报
stock_start = stock_date.data.loc['2019-01-02'].reset_index().set_index('code')["close"]
stock_diff = (stock_date.data.loc['2019-06-14'].reset_index().set_index('code')["close"] - stock_start)/stock_start*100

# 计算指数回报
index =QA.QA_fetch_index_day_adv('000001', '2019-01-02', '2019-06-14').data
index_diff = (index.loc['2019-06-14']["close"].values[0] -
              index.loc['2019-01-03']["close"].values[0])/index.loc['2019-01-03']["close"].values[0]*100

# 添加股票名称
result = pd.merge(stock_diff,infos.loc[codes]['name'], left_index=True,right_index=True)


(result['close'] - index_diff).sum() / result['close'].count()
result['close'].sum() / result['close'].count()
result['close'].sum()
result[result['close'] >0].count()
result[result['close']-index_diff >0].count()

result[result['close']<=0].count()