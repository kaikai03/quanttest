# -*- coding: utf-8 -*-
import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QAParameter import MARKET_TYPE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import backtest_base as trade
import minRisk_SVD

from QUANTAXIS.QAData.QADataStruct import QA_DataStruct_Stock_day

pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

pos = ['002415', '002008', '002009', '002011', '002055', '002139', '002236',
       '002241', '002230', '002444', '603660', '601138']
neg = ['002368', '002369', '002178', '000008', '000555', '000676', '000922',
       '002006', '002338', '002526', '002611', '603828', '603633']
codes = pos+neg
codes = QA.QA_fetch_stock_block_adv().get_block(['智能交通','智慧城市',
                                                 '安防服务','人脸识别','机器人概念','电子制造',
                                                 '电器仪表','医疗保健','医疗器械','医疗器械服务',
                                                 '医疗改革','医药','医药商业', '医药电商']).code


 QA.QA_fetch_stock_block_adv().get_block(['医药']).data

 infos = QA.QA_fetch_stock_list_adv()
[(code, infos[infos.code==code].name[0])
 for code in QA.QA_fetch_stock_block_adv().get_block(['医疗器械']).code]



start_date = '2018-01-05'
end_date = '2018-12-30'
pre_dates = 35
init_cash = 1000000
data = QA.QA_fetch_stock_day_adv(codes,
                                 QA.QA_util_get_pre_trade_date(start_date, pre_dates),
                                 end_date).to_hfq()
refresh_interval = 5

account = QA.QA_Account(QA.QA_util_random_with_topic("admin_"),QA.QA_util_random_with_topic("admin_co_"))
broker = QA.QA_BacktestBroker()
account.reset_assets(init_cash)


# t = data.select_time(QA.QA_util_get_pre_trade_date('2019-01-05',30),
#                  QA.QA_util_get_pre_trade_date('2019-01-05',1))
# w = get_weight(t, filt=True)



def get_weight(data_fragment, filt=True):
    lgr_mat, samples = minRisk_SVD.preprocess(data_fragment)
    m = minRisk_SVD.marchenko_pastur_optimize(lgr_mat,samples)
    m.fit()
    if filt:
        return m.filt_min_var_weights_series_norm.apply(np.real)
    else:
        return m.normal_min_var_weights_series_norm.apply(np.real)

def get_trend(data_fragment, args):
    return (args* data_fragment.price_chg).sum()


# account.history_table
# account.hold_available
# account.current_hold_price()
# account.hold_price()

# ws ={}
# for code in data.code:
#     ws[code]=[]

trade_days = 0
for items in data.panel_gen:
    if trade_days < pre_dates:
        trade_days += 1
        continue

    if (trade_days-pre_dates) % refresh_interval != 0:
        trade_days += 1
        continue

    trade_days += 1

    train_data = data.select_time(QA.QA_util_get_pre_trade_date(items.date[0],10),
                 QA.QA_util_get_pre_trade_date(items.date[0],1))
    weights = get_weight(train_data, filt=True)
    weights.index.name = 'code'
    print(QA.QA_util_get_pre_trade_date(items.date[0],10),
               QA.QA_util_get_pre_trade_date(items.date[0],1))

    # train_data = data.select_time(QA.QA_util_get_pre_trade_date(items.date[0],5),
    #              QA.QA_util_get_pre_trade_date(items.date[0],1))
    trend = 1 #get_trend(train_data,account.hold)


    # if trade_days >= 15:break

    if (trade_days - pre_dates) == 1:
        now_asset = np.array(0)
    else:
        now_asset = (account.hold_available*items.open).fillna(0)

    # old_weight = now_asset/now_asset.sum()

    weights =weights.loc[~(weights<0.01)]
    if trend >= 0:
        weights=(weights)/((weights).sum())
    else:
        weights=(1/weights)/((1/weights).sum())

    #只选最大与最小的两个
    if (trade_days - pre_dates) != 1:
        # weights = weights.loc[(weights==weights.max()) ]#|  (weights==weights.min()) ]
        weights = weights.sort_values(ascending=False).iloc[0:1]

    new_weight = (weights+items.open*0).fillna(0)
    new_asset = (now_asset.sum()+account.cash_available)*0.9*new_weight

    # if len(new_weight) == 25:
    #     for c, w in new_weight.iteritems():
    #         ws[c[1]].append(w)

    if (trade_days -pre_dates) == 1:
        change_asset = new_asset
    else:
        change_asset = new_asset-now_asset

    change_asset = change_asset[change_asset.index.levels[0][0],:].sort_values()

    # print(weights)
    # print(change_asset)

    for code, change in change_asset.iteritems():
        print(code,change)
        if change == 0:
            continue

        data_df = items.data.xs(code,level=1,drop_level=False)
        item = QA_DataStruct_Stock_day(data_df, dtype=items.type,  if_fq=items.if_fq)

        if change < 0:
            mount = abs(change)/item.open[0]
            if mount < 100 : continue
            mount = int(mount/100)*100
            trade.sell_item_mount(account, broker, item, mount)

        if change > 0:
            trade.buy_item_money(account, broker, item,
                         change,
                         item.open[0])

        account.settle()





for item in data.security_gen:
    end_item = item.select_time(item.date[-1],item.date[-1])
    sell_unit = account.sell_available.get(item.code,0)
    if sell_unit[0] > 0:
        trade.sell_item_mount(account, broker, end_item, sell_unit[0])
        account.settle()

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

account.history_table
data.data.loc[['2018-05-10','2018-05-11']].pct_change()

account.history_table.amount.plot()
