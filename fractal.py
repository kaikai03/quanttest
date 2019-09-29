# -*- coding: utf-8 -*-
# Name: fractal strategy

import QUANTAXIS as QA
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import datetime
import matplotlib.finance as mpf

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 300)


code = '002415'
start = '2018-03-03'
end = '2018-06-30'
colors = {-1:"green",1:"pink",0:"grey"}
data = QA.QA_fetch_stock_day_adv(code, start, end).to_hfq()
# data = QA.QA_fetch_stock_min_adv(code, start, end,frequence='5min').to_qfq().reset_index()


fig = plt.figure('', figsize=(12,6))
fig.subplots_adjust(bottom=0.1)
ax1  = fig.add_subplot(211)

ax1.bar(data.date.values,(data.high-data.low).values,
        bottom=data.low.values,
        color=[colors[i] for i in np.sign(data.close-data.open)])

ax1.scatter(data[data.close<data.open].date.values, data[data.close<data.open].low*0.99,marker='v',c="g")
ax1.scatter(data[data.close>data.open].date.values, data[data.close>data.open].high*1.01,marker='^',c="pink")


quotes = data[['open','high','low','close']].data.reset_index()
del quotes["code"]
quotes['date'] = quotes['date'].apply(lambda x: date2num(x))

ax2  = fig.add_subplot(212)
ax2.xaxis_date()
mpf.candlestick_ohlc(ax2, quotes.values, width=0.7, colorup='r', colordown='g')

for i in range(quotes.shape[0]):
    if i < 5:continue
    quotes.loc[i].high
    cur_lh = (quotes.loc[i].low,quotes.loc[i].high)
    up_arrow = -1 # 1 up 2 down -1 break
    left_chk = False
    last_lh = None
    for left_step in range(1,5):
        left_lh = (quotes.loc[i-left_step].low,quotes.loc[i-left_step].high)
        if up_arrow != -1:
            left_lh[0] > last_lh[0] and left_lh[1] > last_lh[1]:

        if left_lh[0] >= cur_lh[0] and left_lh[1] <= cur_lh[1]:
            continue
        if left_lh[0] > cur_lh[0] and left_lh[1] > cur_lh[1]:
            up_arrow = 2
            last_lh = left_lh
            continue
        if left_lh[0] < cur_lh[0] and left_lh[1] < cur_lh[1]:
            up_arrow = 1
            last_lh = left_lh
            continue

    if left_chk:
        print(i)