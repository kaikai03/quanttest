import QUANTAXIS as QA

def deal_order(account, broker, order,item):
    broker.receive_order(QA.QA_Event(order=order, market_data=item))
    trade_mes = broker.query_orders(account.account_cookie, 'filled')
    res = trade_mes.loc[order.account_cookie, order.realorder_id]
    order.trade(res.trade_id, res.trade_price,
                res.trade_amount, res.trade_time)


def buy_item_mount(account, broker, item, amount):
    order = account.send_order(
        code=item.code[0],
        time=item.date[0],
        amount=amount,
        towards=QA.ORDER_DIRECTION.BUY,
        order_model=QA.ORDER_MODEL.CLOSE,
        amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
    )
    deal_order(account, broker, order, item)

def buy_item_money(account, broker, item, money, price):
    order = account.send_order(
        code=item.code[0],
        time=item.date[0],
        money=money,
        price=price,
        towards=QA.ORDER_DIRECTION.BUY,
        order_model=QA.ORDER_MODEL.CLOSE,
        amount_model=QA.AMOUNT_MODEL.BY_MONEY
    )
    deal_order(account, broker, order, item)


def sell_item_mount(account, broker, item, amount):
    order = account.send_order(
        code=item.code[0],
        time=item.date[0],
        amount=amount,
        towards=QA.ORDER_DIRECTION.SELL,
        price=0,
        order_model=QA.ORDER_MODEL.MARKET,
        amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
    )
    deal_order(account, broker, order, item)