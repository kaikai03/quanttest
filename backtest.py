# coding=utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2017 yutiansut/QUANTAXIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import  matplotlib
import QUANTAXIS as QA
from QUANTAXIS import QA_Backtest as QB
import pandas as pd
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 600)
import numpy as np

import analysis_tools as at


"""
写在前面:
===============QUANTAXIS BACKTEST STOCK_DAY中的变量
常量:
QB.backtest_type 回测类型 day/1min/5min/15min/30min/60min/index_day/index_1min/index_5min/index_15min/index_30min/index_60min/
QB.account.message  当前账户消息
QB.account.cash  当前可用资金
QB.account.hold  当前账户持仓
QB.account.history  当前账户的历史交易记录
QB.account.assets 当前账户总资产
QB.account.detail 当前账户的交易对账单
QB.account.init_assest 账户的最初资金
QB.strategy_gap 前推日期
QB.strategy_name 策略名称

QB.strategy_stock_list 回测初始化的时候  输入的一个回测标的
QB.strategy_start_date 回测的开始时间
QB.strategy_end_date  回测的结束时间

QB.setting.QA_setting_user_name = str('admin') #回测账户
QB.setting.QA_setting_user_password = str('admin') #回测密码

QB.today  在策略里面代表策略执行时的日期
QB.now  在策略里面代表策略执行时的时间
QB.benchmark_code  策略业绩评价的对照行情
QB.benchmark_type  对照行情是股票还是指数


QB.backtest_print_log = True  # 是否在屏幕上输出结果

函数:
获取市场(基于gap)行情:
QB.QA_backtest_get_market_data(QB,code,QB.today)
获取单个bar
QB.QA_backtest_get_market_data_bar(QB,code,QB.today/QB.now)

拿到开高收低量
Open,High,Low,Close,Volume=QB.QA_backtest_get_OHLCV(QB,QB.QA_backtest_get_market_data(QB,item,QB.today))

获取市场自定义时间段行情:
QA.QA_fetch_stock_day(code,start,end,model)

一键平仓:
QB.QA_backtest_sell_all(QB)

报单:
QB.QA_backtest_send_order(QB, code,amount,towards,order: dict)

order有三种方式:
1.限价成交 order['bid_model']=0或者l,L
  注意: 限价成交需要给出价格:
  order['price']=xxxx

2.市价成交 order['bid_model']=1或者m,M,market,Market  [其实是以bar的开盘价成交]
3.严格成交模式 order['bid_model']=2或者s,S
    及 买入按bar的最高价成交 卖出按bar的最低价成交
3.收盘价成交模式 order['bid_model']=3或者c,C

#查询当前一只股票的持仓量
QB.QA_backtest_hold_amount(QB,code)
#查询当前一只股票的可卖数量
QB.QA_backtest_sell_available(QB,code)
#查询当前一只股票的持仓平均成本
QB.QA_backtest_hold_price(QB,code)
"""


@QB.backtest_init
def init():

    # 回测的类别
    # day/1min/5min/15min/30min/60min/index_day/index_1min/index_5min/index_15min/index_30min/index_60min/
    QB.backtest_type = 'day'
    # QB.backtest_type='5min' # 日线回测
    # 策略的名称
    QB.strategy_name = 'test_daily'
    # 数据库位置
    QB.setting.QA_util_sql_mongo_ip = '127.0.0.1'  # 回测数据库
    QB.setting.QA_setting_user_name = str('admin') #回测账户
    QB.setting.QA_setting_user_password = str('admin') #回测密码

    QB.account.init_assest = 2500000  # 初始资金

    # benchmark
    # QB.benchmark_code = '399300'
    QB.benchmark_code = '600018'
    # benchmark 可以是个股，也可以是指数
    # QB.benchmark_type = 'index'
    QB.benchmark_type = 'stock'
    # 手续费系数
    QB.commission_fee_coeff = 0.0015  # 千五的手续费(单向)

    QB.strategy_gap = 30   # 在取数据的时候 向前取多少个bar(会按回测的时间动态移动)
    QB.strategy_stock_list = ['600016']  # 回测的股票列表/如果是指数回测 就是指数列表
    QB.strategy_start_date = '2016-07-01 10:30:00'  # 回测开始日期
    QB.strategy_end_date = '2016-08-02'    # 回测结束日期
    QB.backtest_print_log = False  # 是否在屏幕上输出结果



@QB.before_backtest
def before_backtest():
    global risk_position


@QB.load_strategy
def strategy():
    global risk_position  # 在这个地方global变量 可以拿到before_backtest里面的东西
    QA.QA_util_log_info(QB.account.sell_available)
    QA.QA_util_log_info('LEFT Cash: %s' % QB.account.cash_available)
    for item in QB.strategy_stock_list:
        QA.QA_util_log_info(
            QB.QA_backtest_get_market_data(QB, item, QB.today).data)  # 如果是分钟回测 用QB.now

        if QB.QA_backtest_hold_amount(QB, item) == 0:  # 如果不持仓
            QB.QA_backtest_send_order(
                QB, item, 10000, 1, {'bid_model': 'Market'})
        elif QB.QA_backtest_sell_available(QB, item) > 0:  #如果可卖数量大于0
            QB.QA_backtest_send_order(
                QB, item, 10000, -1, {'bid_model': 'Market'})
                
# #查询当前一只股票的持仓量
# QB.QA_backtest_hold_amount(QB,code)
# #查询当前一只股票的可卖数量
# QB.QA_backtest_sell_available(QB,code)
# #查询当前一只股票的持仓平均成本
# QB.QA_backtest_hold_price(QB,code)
@QB.end_backtest
def after_backtest():
    pass






def baseData():
    #函数演示
    QA.QA_fetch_stock_day_adv('000001','2017-09-01','2017-10-01').pivot('close')
    QA.QA_fetch_stock_day_adv('000001','2017-09-01','2017-10-01').get_bar('000001','2017-09-05', True).data
    QA.QA_fetch_stock_day_adv('000001','2017-09-01','2017-10-01').plot()  # .query('volume>1')
    QA.QA_fetch_stock_day_adv('000001','2017-09-01','2017-10-01').to_hfq().data
    QA.QA_fetch_stock_day_adv('000001','2017-09-01','2017-10-01').to_hfq().plot()

    days =QA.QA_fetch_stock_day_adv(['000001'],'2016-01-01','2017-12-30').data
    # days =QA.QA_fetch_index_day_adv(['000001','000004','000006'],'2018-01-01','2018-12-30').data
    ret =  (days[days["code"]=="000001"]["close"]/days[days["code"]=="000001"]["close"].shift(1)).fillna(0)

    ret_funtion = lambda rows : (rows[["open", "high", "low", "close"]]-rows[["open", "high", "low", "close"]].shift(1)).fillna(0)
    days[["open", "high", "low", "close"]] = days.groupby("code").apply(ret_funtion)

    ret_r_funtion = lambda rows : ((rows[["open", "high", "low", "close"]]/rows[["open", "high", "low", "close"]].shift(1)-1)).fillna(0)
    ## 对数收益率 或 复合回报率
    ret_log_funtion = lambda rows : np.log((rows[["open", "high", "low", "close"]]/rows[["open", "high", "low", "close"]].shift(1))).fillna(0)
    days[["open", "high", "low", "close"]] = days.groupby("code").apply(ret_log_funtion)
    at.drawplot_df(days)

    days.query('close>9.39')
    days.index
    ß
    at.drawplot_df(days,split=True)
    at.drawplot_df(days)
    webbrowser.open('file://'+os.path.realpath('QA_stock_day_None_hfq.html'))


    s0001 = days[days.index.get_level_values("code")=="000001"]
    s0001[["open", "high", "low", "close"]] = s0001.groupby("code").apply(ret_funtion)
    s0001[["open", "high", "low", "close"]] = s0001.groupby("code").apply(ret_r_funtion)
    s0001[["open", "high", "low", "close"]] = s0001.groupby("code").apply(ret_log_funtion)


    s0001["close"] =  np.random.normal(0., 0.1, s0001.shape[0])
    s0001["close"] =  np.random.rand(s0001.shape[0])
    s0001["close"] =  np.random.standard_exponential(s0001.shape)

    at.drawplot_df(s0001)
    at.hurst_RS(s0001)
    at.hurst(s0001["close"])
    pass