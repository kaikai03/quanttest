import QUANTAXIS as QA
import random
import time
import numpy as np
import pandas as pd
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 500)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

import backtest_base as trade
import minRisk_SVD as SVD

from multiprocessing.dummy import Pool

class myError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

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

    data = QA.QA_fetch_stock_day_adv(weights.index.to_list(), start_date, end_date)
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

def count_continuous(sign_Series):
    max_continuous_positive_count=0
    max_continuous_negative_count=0
    continuous_tem = 0
    for idx,item in enumerate(sign_Series):
        if idx==0:continue
        if item==0:continue
        if item != sign_Series.iloc[idx-1]:
            continuous_tem=0
            continue

        continuous_tem+=1
        if item > 0:
            max_continuous_positive_count = max(max_continuous_positive_count,continuous_tem)
        else:
            max_continuous_negative_count = max(max_continuous_negative_count,continuous_tem)

    return max_continuous_positive_count,max_continuous_negative_count

def cross_describe(codes, start='2010-01-01',end='2018-12-30', bench=False):
    #code = "000001",start='2015-01-01',end='2018-12-30'
    st_time = time.time()

    global items
    items = []

    def deal(code):
        print(" process:",code)
        stock_name = None

        if not str(code)[0] in ['0','6']:
            print("jump:", code)
            return

        data = QA.QA_fetch_stock_day_adv(code,start,end)
        if data is None or len(data)==0:
            print("jump:", code)
            return
        data = data.to_qfq()
        if code in code_in_infos:
            stock_name = stock_infos.loc[code]['name']

        years_list = data.date.year.value_counts()
        # year = years_list.index[0]

        tmp = []
        for year in years_list.index:
            open_days = years_list[year]
            period_data = data.select_time(str(year),str(year+1))
            if len(period_data) == 0:
                continue

            priod_pct = period_data.close_pct_change()
            priod_pct_shift = priod_pct.shift(1)
            sig = np.sign(priod_pct)
            up_down_count = sig.value_counts()

            compare = np.sign(priod_pct) == np.sign(priod_pct_shift)
            cross = compare.value_counts()[False]

            co_pos,cp_neg = count_continuous(sig)

            # "days":open_days,  "cross":cross,
            item={"code":code, "name":stock_name, "year":year,
             "co_p":co_pos, "co_n":cp_neg, "cros_r":round(cross/open_days,2),
             "go_up":up_down_count.get(1,None),"go_dn":up_down_count.get(-1,None),
             "max_pri":round(period_data.close.max(),2),"min_pri":round(period_data.close.min(),2)}

            tmp.append(item)
        items.extend(tmp)

    stock_infos = QA.QA_fetch_stock_list_adv()
    code_in_infos = stock_infos.code
    size = len(codes)

    pool = Pool(4)
    pool.map(deal, codes)
    pool.close()
    pool.join()
    # if bench:
    #     codes.insert(0,'000001')
    #         if bench and idx == 0:
    #         data = QA.QA_fetch_index_day_adv(code,start,end)
    #         stock_name = 'bench'
    #     else:

    # for idx, code in enumerate(codes):


    df = pd.DataFrame(items)
    df.set_index(["code","name","year"], inplace=True)
    print("used:",(time.time()-st_time))
    return df

len(codes.code)
df = cross_describe(codes.code, bench=True)
df
df.sort_index().to_excel('./file/cross_describe.xls')




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