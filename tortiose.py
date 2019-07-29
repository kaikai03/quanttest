# -*- coding: utf-8 -*-
import QUANTAXIS as QA
import pandas as pd

import matplotlib.pyplot as plt


import backtest_base as trade

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

################### init_variable
code_list = ['000001', '000002', '000004', '600000',
             '600536', '000936', '002023', '600332',
             '600398', '300498', '603609', '300673']
code_list = QA.QA_fetch_stock_block_adv().code[0:2]

code = '002511'
start_date = '2017-01-01'
end_date = '2017-12-31'
init_cash = 100000

################strategy
def ATR_strategy(data, duration=20):
    tr = QA.MAX(QA.MAX(data.high - data.low, data.high - data.preclose),data.preclose - data.low)
    atr = QA.MA(tr, duration)
    return pd.DataFrame({'TR': tr, 'ATR': atr})

################### base deal
resualts = []
def multi_test(codes, date_ranges):
    for code in codes:
        for date_range in date_ranges:
            resualt = tortiose_one(code, date_range)
            resualts.append((code,date_range,resualt['a_r'],resualt['bm_ar'],resualt['alpha']))

multi_test(code_list, [('2019-01-01','2019-12-31'),('2018-01-01','2018-12-31'),
                       ('2017-01-01','2017-12-31'),('2016-01-01','2016-12-31')])



#########################base
def tortiose_one(code,date_range,init_cash=100000):
    account = QA.QA_Account(QA.QA_util_random_with_topic("admin_"),QA.QA_util_random_with_topic("admin_co_"))
    broker = QA.QA_BacktestBroker()
    account.reset_assets(init_cash)

    start_date = date_range[0]
    end_date = date_range[1]

    data = QA.QA_fetch_stock_day_adv(code, start_date, end_date)
    print(data)
    if data == None:
        return {"bm_ar":None,"alpha":None,'dropback':None,'beta':None,'sharpe':None, 'a_r':None,
            'risk':None, "account":None}
    backtest_data = data.to_qfq()
    # backtest_data.plot()


    ind=backtest_data.add_func(ATR_strategy)
    inc=QA.QA_DataStruct_Indicators(ind)
    # inc.get_code(code).atr.plot()

    up_line = QA.HHV(backtest_data.price.shift(1),20)
    down_line = QA.LLV(backtest_data.price.shift(1),10)

    # backtest_data.price.plot(label="pri")
    # up_line.plot(label="up")
    # down_line.plot(label="down")
    # plt.legend(loc=0)
    ################run
    lastATR = 0
    for items in backtest_data.panel_gen:
        for item in items.security_gen:
            up = up_line.loc[item.date[0],item.code[0]]
            atr = inc.get_indicator(item.date[0],code,'ATR')
            if pd.isna(atr):
                continue

            cur = item.price[0]
            need_buy = False
            if item.price[0] > up:
                if len(account.history_table)>0:
                    if account.history_table.iloc[-1].amount<=0:
                        need_buy = True
                elif len(account.history_table)==0:
                    need_buy = True

            if len(account.history_table) > 0:
                if item.price[0] > account.history_table.iloc[-1].price + 0.5 * lastATR and account.history_table.iloc[
                    -1].amount > 0 :
                    need_buy = True


            if need_buy:
                buy_unit = int((0.01 * account.cash_available) / atr)
                if buy_unit < 100:
                    continue

                if buy_unit * cur > account.cash_available:
                    buy_unit = int(account.cash_available * 0.1 / cur)

                ret = trade.buy_item_mount(account, broker, item, buy_unit, cur)
                if ret:
                    account.settle()
                    lastATR = atr
                continue


            if item.price[0] < down_line.loc[item.date[0],item.code[0]]:
                sell_unit = account.sell_available.get(code,0)
                if sell_unit <= 0:
                    continue
                ret = trade.sell_item_mount(account, broker, item, sell_unit)
                if ret:
                    account.settle()
                    lastATR = 0

            if len(account.history_table) > 0:
                if item.price[0] < account.history_table.iloc[-1].price - 0.2 * lastATR and account.history_table.iloc[
                    -1].amount > 0 :
                    sell_unit = account.sell_available.get(code, 0)
                    if sell_unit <= 0:
                        continue
                    ret = trade.sell_item_mount(account, broker, item, sell_unit)
                    if ret:
                        account.settle()
                        lastATR = 0
                    continue


    end_item =  data.select_time(data.date[-1],data.date[-1])
    sell_unit = account.sell_available.get(code,0)
    if sell_unit > 0:
        trade.sell_item_mount(account, broker, end_item, sell_unit)
        account.settle()

    risk = QA.QA_Risk(account)
    bm_ar = risk.message['bm_annualizereturn']
    dropback = risk.message['max_dropback']
    a_r = risk.message['annualize_return']
    sharpe = risk.message['sharpe']
    beta = risk.message['beta']
    alpha = risk.message['alpha']
    return {"bm_ar":bm_ar,"alpha":alpha,'dropback':dropback,'beta':beta,'sharpe':sharpe, 'a_r':a_r,
            'risk':risk, "account":account}


print(account.history)
print(account.history_table)
print(account.daily_hold)

# create Risk analysis
Risk = QA.QA_Risk(account)
print(Risk.message)
print(Risk.assets)
Risk.market_data.data

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

pr=QA.QA_Performance(account)
pr.pnl_fifo
pr.plot_pnlmoney()
pr.plot_pnlratio()
pr.message