import QUANTAXIS as QA
import random

market = QA.QA_Market()
user = QA.QA_Portfolio('user_admin')

Account = user.new_account('user_admin_qsdd')

market.start()
market.connect(QA.RUNNING_ENVIRONMENT.BACKETEST)

market.login(QA.BROKER_TYPE.BACKETEST, Account.account_cookie, Account)

for date in QA.QA_util_get_trade_range('2017-01-01','2017-01-31'):
    for code in ['000001', '000002', '000004', '000007']:
        if random.random()<0.3:
            market.insert_order(account_cookie=Account.account_cookie, amount=1000, price=None, amount_model=QA.AMOUNT_MODEL.BY_AMOUNT, time=date, code=code,
                                order_model=QA.ORDER_MODEL.CLOSE, towards=QA.ORDER_DIRECTION.BUY, market_type=QA.MARKET_TYPE.STOCK_CN,
                                frequence=QA.FREQUENCE.DAY, broker_name=QA.BROKER_TYPE.BACKETEST)
        else:
            try:
                print(user.get_account(Account.account_cookie).sell_available.get(code,0))
                market.insert_order(account_cookie=Account.account_cookie, amount=1000, price=None, amount_model=QA.AMOUNT_MODEL.BY_AMOUNT, time=date, code=code,
                                    order_model=QA.ORDER_MODEL.CLOSE, towards=QA.ORDER_DIRECTION.SELL, market_type=QA.MARKET_TYPE.STOCK_CN,
                                    frequence=QA.FREQUENCE.DAY, broker_name=QA.BROKER_TYPE.BACKETEST)
            except:
                pass
    print("deal:", date, code)
    # market._settle(QA.BROKER_TYPE.BACKETEST)
    market.sync_order_and_deal()


market.order_handler.order_queue()
market.order_handler.order_queue.order_list
market.order_handler.order_queue.pending