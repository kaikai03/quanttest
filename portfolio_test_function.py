import QUANTAXIS as QA
import random
import time
import numpy as np
import pandas as pd


import backtest_base as trade
import minRisk_SVD as SVD



def empty_assets(account, broker, holded_start_date, holded_end_date):
    for code in account.sell_available.index:
        sell_availiable=account.sell_available[code]

        data = QA.QA_fetch_stock_day_adv(code, holded_start_date, holded_end_date)
        data = data.to_qfq()

        end_item =  data.select_time(data.date[-1],data.date[-1])
        if sell_availiable <=end_item.close[0]:
            continue
        trade.sell_item_mount(account, broker, end_item, sell_availiable)
        account.settle()


def hold_test(weights, start_date, end_date, init_cash=10000000):
    assert type(weights) == pd.Series
    assert weights.sum() >0.999999

    account = QA.QA_Account(QA.QA_util_random_with_topic("admin_"),QA.QA_util_random_with_topic("admin_co_"))
    broker = QA.QA_BacktestBroker()
    account.reset_assets(init_cash)

    data = QA.QA_fetch_stock_day_adv(list(weights.index), start_date, end_date)
    data = data.to_qfq()
    if data is None:return None
    if len(data) == 0:return None

    start_item =  data.select_time(data.date[1],data.date[1])
    cash_scale = weights * init_cash

    if len(weights)==1:
        cash_scale = cash_scale*0.94

    for item in start_item.security_gen:
        if cash_scale[item.code[0]] < item.close.values[0]*100:
            continue
        print(item.code[0], round(cash_scale[item.code[0]],2),item.close.values[0])
        ret = trade.buy_item_money(account, broker, item, round(cash_scale[item.code[0]],2),item.close.values[0])
        if ret:
            account.settle()

    empty_assets(account, broker, start_date, end_date)

    return account






##########################################test###############
init_cash = 1000000
start_date = '2017-01-08'
end_date = '2017-12-26'
QA.QA_fetch_stock_block_adv().data.index.levels[0]
codes = QA.QA_fetch_stock_block_adv().get_block(['生物医药','化学制药']).code
codes[1:10]
codes = ["002415",'000900','600352','600759','601600','600392','600523','600139','600885','600575','600975',
         '600586','601118','600738','603648','600426','600217','600483','601669','603008','600371','600965',
         '600377','601988','601009','600639','600145','600681','600598','603690','600535','600030',
         '600257','601607','600926','603799','600084','600763','603978']
#QA.QA_util_get_trade_range(start_date, end_date)
####################
codes = QA.QA_fetch_stock_block_adv().get_block(['生物医药','化学制药']).code
data =QA.QA_fetch_stock_day_adv(codes[1:10],start_date,end_date).to_qfq()

data =QA.QA_fetch_stock_day_adv(codes,start_date,end_date).to_qfq()
lgR_mat,samples = SVD.preprocess(data)
m = SVD.marchenko_pastur_optimize(lgR_mat,samples)
m.fit()

weights__ = pd.Series([0.5,0.3,0.2], ["000001","000002","000004"])
list(weights__.index)
start_date = '2018-01-08'
end_date = '2018-12-26'
weights__ = m.filt_min_var_weights_series_norm
weights2__ = m.normal_min_var_weights_series_norm
Account = hold_test(weights__, start_date, end_date)
Account2 = hold_test(weights2__, start_date, end_date)
Risk = QA.QA_Risk(Account)
Risk2 = QA.QA_Risk(Account2)
fig = Risk.assets.plot()
fig = Risk2.assets.plot()
fig.legend(["A",'B'])
###################


tmp = pd.Series()
tmp_ = pd.Series()
for i,code in enumerate(codes):
    Account = hold_test(pd.Series([1], [code]), start_date, end_date)
    if Account is None or len(Account.history)==0:
        continue

    Risk = QA.QA_Risk(Account)
    tmp[code] = Risk.message["annualize_return"]
    tmp_[code] = Risk.message["max_dropback"]
    print(i,'/',len(codes),"codes:",code,":",Risk.message["annualize_return"])

df = pd.DataFrame(list(zip(tmp, tmp_)),index=tmp.index,columns=["re","drop"])
df.shape
df.re

(df.re<0).value_counts()
df.re.sum()
df.re[df.re<0].sum()
df.re[df.re>=0].sum()
df.re[df.re<0].min()
df.re[df.re>=0].max()



Account = hold_test(pd.Series([1], ["002415"]), start_date, end_date)

print(Account.history)
print(Account.history_table)
print(Account.daily_hold)

# create Risk analysis
Risk = QA.QA_Risk(Account)
print(Risk.message)
print(Risk.assets["2018-01-09"])
Risk.message["annualize_return"]
Risk.daily_market_value
Account.daily_hold.apply(abs)



fig = Risk.assets.plot()
fig.plot()
# Risk.benchmark_assets.plot()
fig = Risk.plot_assets_curve()
fig.plot()
fig = Risk.plot_dailyhold()
fig.plot()
fig = Risk.plot_signal()
fig.plot()

plt.show()

pr=QA.QA_Performance(Account)
pr.pnl_fifo
pr.plot_pnlmoney()
pr.plot_pnlratio()
pr.message

cs=QA.QA_fetch_stock_list_adv().code

summ = 0
for code in cs:

    data =QA.QA_fetch_stock_day_adv(code,"2018-05-10","2018-06-15")
    if data is None:continue
    if len(data)==0:continue
    last =  data.select_time(data.date[-1],data.date[-1])
    summ += last.close[0]*100
    print(code, last.close[0]*100)

summ