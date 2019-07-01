import QUANTAXIS as QA
import random
import pandas as pd
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 500)

market = QA.QA_Market()
user = QA.QA_Portfolio('user_admin')

Account = user.new_account('user_admin_qsdd')
Account.tax_coeff = 0.0015
Account.commission_coeff=0.00025

market.start()
market.connect(QA.RUNNING_ENVIRONMENT.BACKETEST)

market.login(QA.BROKER_TYPE.BACKETEST, Account.account_cookie, Account)

for date in QA.QA_util_get_trade_range('2017-01-01','2017-01-31'):
    for code in ['000001', '000002', '000004', '000007']:
        if random.random()<0.3:
            try:
                market.insert_order(account_cookie=Account.account_cookie, amount=1000, price=None, amount_model=QA.AMOUNT_MODEL.BY_AMOUNT, time=date, code=code,
                                    order_model=QA.ORDER_MODEL.CLOSE, towards=QA.ORDER_DIRECTION.BUY, market_type=QA.MARKET_TYPE.STOCK_CN,
                                    frequence=QA.FREQUENCE.DAY, broker_name=QA.BROKER_TYPE.BACKETEST)
            except:
                print("error: 交易失败 跳过", date, code)
                continue
        else:
            try:
                if(Account.sell_available.get(code,0)<=0):
                    print("未持有")
                    continue
                market.insert_order(account_cookie=Account.account_cookie, amount=1000, price=None, amount_model=QA.AMOUNT_MODEL.BY_AMOUNT, time=date, code=code,
                                    order_model=QA.ORDER_MODEL.CLOSE, towards=QA.ORDER_DIRECTION.SELL, market_type=QA.MARKET_TYPE.STOCK_CN,
                                    frequence=QA.FREQUENCE.DAY, broker_name=QA.BROKER_TYPE.BACKETEST)
            except:
                print("error: 交易失败 跳过", date, code)
                continue
    print("deal:", date, code)
    market._settle(QA.BROKER_TYPE.BACKETEST)
    market.sync_order_and_deal() ## 实盘时不会实时返回订单信息，靠2秒一次轮询


market.order_handler.order_queue()
market.order_handler.order_queue.order_list
market.order_handler.order_queue.pending
market.get_trading_day()


Account.sell_available

Risk = QA.QA_Risk(Account)
print(Risk.message)
fig = Risk.assets.plot()
fig.plot()
# Risk.benchmark_assets.plot()
fig = Risk.plot_assets_curve()
fig.plot()
fig = Risk.plot_dailyhold()
fig.plot()
fig = Risk.plot_signal()
fig.plot()

pr=QA.QA_Performance(Account)
pr.pnl_fifo
pr.plot_pnlmoney()
pr.plot_pnlratio()
pr.message