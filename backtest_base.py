import QUANTAXIS as QA

def deal_order(account, broker, order,item):
    broker.receive_order(QA.QA_Event(order=order, market_data=item))
    trade_mes = broker.query_orders(account.account_cookie, 'filled')
    res = trade_mes.loc[order.account_cookie, order.realorder_id]
    order.trade(res.trade_id, res.trade_price,
                res.trade_amount, res.trade_time)


def buy_item_mount(account, broker, item, amount, price):
    order = account.send_order(
        code=item.code[0],
        time=item.date[0],
        price=price,
        amount=amount,
        towards=QA.ORDER_DIRECTION.BUY,
        order_model=QA.ORDER_MODEL.LIMIT,
        amount_model=QA.AMOUNT_MODEL.BY_AMOUNT
    )
    if not order:
        return False
    deal_order(account, broker, order, item)
    return True

def buy_item_money(account, broker, item, money, price):
    # if item.close.values[0]*100*(1+account.commission_coeff) >= money:
    #     return False
    order = account.send_order(
        code=item.code[0],
        time=item.date[0],
        money=money,
        price=price,
        towards=QA.ORDER_DIRECTION.BUY,
        order_model=QA.ORDER_MODEL.LIMIT,
        amount_model=QA.AMOUNT_MODEL.BY_MONEY
    )
    if not order:
        return False
    deal_order(account, broker, order, item)
    return True

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
    if not order:
        return False
    deal_order(account, broker, order, item)
    return True