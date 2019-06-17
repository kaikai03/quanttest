
import QUANTAXIS as QA

import pandas as pd
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import matplotlib.pyplot as plt


data =QA.QA_fetch_stock_day_adv(['000001', '000002'],'2018-05-09','2018-05-20')
data_forbacktest=data.select_time('2018-05-09','2018-05-18')#.data
items = data_forbacktest.panel_gen.send(None) #基于日面板迭代器
item = items.security_gen.send(None) #基于代码的迭代器
item.open[0]
item.code[0]

 # 初始化一个account
Account=QA.QA_Account("test_user_cookie", "test_portfolio_cookie")

# 初始化一个回测类
B = QA.QA_BacktestBroker()


# 全仓买入'000001'

order=Account.send_order(code='000001',
                        price=11,
                        money=Account.cash_available,
                        time='2018-05-09',
                        towards=QA.ORDER_DIRECTION.BUY,
                        order_model=QA.ORDER_MODEL.MARKET,
                        amount_model=QA.AMOUNT_MODEL.BY_MONEY
                        )

order=Account.send_order(code='000001',
                        price=11,
                        amount=Account.sell_available.get('000001', 0),
                        time='2018-05-11',
                        towards=QA.ORDER_DIRECTION.SELL,
                        order_model=QA.ORDER_MODEL.MARKET,
                        amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
                        )

## 打印order的占用资金
print('ORDER的占用资金: {}'.format((order.amount*order.price)*(1+Account.commission_coeff)))

# 账户可用资金
print('账户可用资金 :{}'.format(Account.cash_available))

B.receive_order(QA.QA_Event(order=order))
trade_mes = B.query_orders(Account.account_cookie,'filled')
res = trade_mes.loc[order.account_cookie, order.realorder_id]
print(res)
order.trade(res.trade_id,res.trade_price,res.trade_amount,res.trade_time)

Account.settle() ## 结算，数据同步账户

Account.cash_available
Account.hold_available
Account.hold['000001']
Account.history
Account.history_table
Account.cash
Account.daily_cash.cash.plot()
Account.daily_hold.index.levels[0]
Account.running_time
Account.close_positions_order # T0时才有
Account.hold_price()

plt.show()


Risk=QA.QA_Risk(Account)
Risk.message
Risk.market_value.diff().iloc[-1]
Risk.account.cash_table
Risk.market_value.sum(axis=1)

Risk.assets.plot()
# 同上，有时候由于时间问题无法绘图
pd.Series(data=Risk.assets,index=pd.to_datetime(Risk.assets.index),name='date').plot()

Risk.plot_assets_curve()
Risk.plot_dailyhold()
Risk.plot_signal()
Risk.profit_construct
Risk.beta
Risk.alpha
Risk.sharpe

Performance=QA.QA_Performance(Account)
Performance.pnl_fifo
Performance.plot_pnlmoney(Performance.pnl_fifo)

# Account.save()
# Risk.save()

#历史查看
# account_info=QA.QA_fetch_account({'account_cookie':'test_user_cookie'})
# account=QA.QA_Account().from_message(account_info[0])


