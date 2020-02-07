# -*- coding: utf-8 -*-
import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QAParameter import MARKET_TYPE
import pandas as pd

import matplotlib.pyplot as plt


import backtest_base as trade
import minRisk_SVD

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

pos = ['002415', '002008', '002009', '002011', '002055', '002139', '002236',
       '002241', '002230', '002444', '603660'] #, '601138'
neg = ['002368', '002369', '002178', '000008', '000555', '000676', '000922',
       '002006', '002338', '002526', '002611', '603633'] #, '603828'
codes = pos+neg
start_date = '2018-01-15'
end_date = '2018-12-30'
init_cash = 1000000
data = QA.QA_fetch_stock_day_adv(codes, start_date, end_date).to_hfq()


account = QA.QA_Account(QA.QA_util_random_with_topic("admin_"),QA.QA_util_random_with_topic("admin_co_"))
broker = QA.QA_BacktestBroker()
account.reset_assets(init_cash)

# average_cash = init_cash / len(codes)

# weight = data.select_time(data.date[0],data.date[0]).open/data.select_time(data.date[0],data.date[0]).open.sum()
# average_cash = init_cash*.95*weight
# average_cash = average_cash.loc[data.date[0]]

train_start = '2018-01-15'
train_end = '2018-01-30'
train_data = QA.QA_fetch_stock_day_adv(codes, train_start, train_end).to_hfq()

lgR_mat,samples = minRisk_SVD.preprocess(train_data)
m = minRisk_SVD.marchenko_pastur_optimize(lgR_mat,samples)
m.fit()
# m.filt_min_var_weights_series_norm
# m.normal_min_var_weights_series_norm

average_cash = init_cash*0.999*m.normal_min_var_weights_series_norm
average_cash = init_cash*0.999*m.filt_min_var_weights_series_norm


for item in data.security_gen:
    print(item.code[0], item.date[0], item.date[-1])
    if not (item.code[0] in average_cash.index.values):
        continue
    trade.buy_item_money(account, broker, item,
                         average_cash.loc[item.code[0]],
                         item.open[0])
    account.settle()

for item in data.security_gen:
    end_item = item.select_time(item.date[-1],item.date[-1])
    sell_unit = account.sell_available.get(item.code,0)
    if sell_unit[0] > 0:
        trade.sell_item_mount(account, broker, end_item, sell_unit[0])
        account.settle()

account.history_table
risk = QA.QA_Risk(account)
risk.plot_assets_curve()

print(risk.message)
print(risk.assets)

print(risk.message['timeindex'],'\n',
risk.message['profit'],'\n',
risk.message['bm_profit'],'\n',
risk.message['alpha'],'\n',
risk.message['beta'],'\n',
risk.message['volatility'],'\n',
risk.message['max_dropback'],'\n',
risk.message['sharpe'])