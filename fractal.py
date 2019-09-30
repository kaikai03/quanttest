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
start = '2019-06-03'
end = '2019-09-30'
colors = {-1:"green",1:"pink",0:"grey"}
data = QA.QA_fetch_stock_day_adv(code, start, end).to_hfq()
# data = QA.QA_fetch_stock_min_adv(code, start, end,frequence='5min').to_qfq().reset_index()
quotes = data[['open','high','low','close']].data.reset_index()
del quotes["code"]
quotes['date'] = quotes['date'].apply(lambda x: date2num(x))

fig = plt.figure('', figsize=(14,6))
fig.subplots_adjust(bottom=0.1)
ax1  = fig.add_subplot(211)

ax1.bar(data.date.values,(data.high-data.low).values,
        bottom=data.low.values,
        color=[colors[i] for i in np.sign(data.close-data.open)])

# ax1.scatter(data[data.close<data.open].date.values, data[data.close<data.open].low*0.99,marker='v',c="g")
# ax1.scatter(data[data.close>data.open].date.values, data[data.close>data.open].high*1.01,marker='^',c="pink")

ax1.scatter(data[down_marks].date.values, data[down_marks].low*0.99,marker='v',c="black")
ax1.scatter(data[up_marks].date.values, data[up_marks].high*1.01,marker='^',c="black")




ax2  = fig.add_subplot(212)
ax2.xaxis_date()
mpf.candlestick_ohlc(ax2, quotes.values, width=0.7, colorup='r', colordown='g')



up_marks=[]
down_marks=[]
for i in range(quotes.shape[0]):
    if i < 5:
        up_marks.append(False);down_marks.append(False);continue

    cur_lh = (quotes.loc[i].low,quotes.loc[i].high)
    left_arrow = -1 # 1 up 2 down -1 break
    right_arrow = -1 # 1 up 2 down -1 break
    left_chk = False
    right_chk = False
    last_lh = None
    for left_step in range(1,5):
        left_lh = (quotes.loc[i-left_step].low,quotes.loc[i-left_step].high)
        if left_arrow != -1:
            if left_lh[0] >= last_lh[0] and left_lh[1] <= last_lh[1]:
                continue

            if left_lh[0] > last_lh[0] and left_lh[1] > last_lh[1]:
                if left_arrow == 2:
                    left_chk = True
                    break
                if left_arrow == 1:
                    break

            if left_lh[0] < last_lh[0] and left_lh[1] < last_lh[1] and left_arrow == 1:
                if left_arrow == 1:
                    left_chk = True
                    break
                if left_arrow == 2:
                    break

        if left_lh[0] >= cur_lh[0] and left_lh[1] <= cur_lh[1]:
            continue  # 左边被包含，直接算下一个
        if left_lh[0] < cur_lh[0] and left_lh[1] > cur_lh[1]:
            break  # 左边整条大于中间，直接退出
        if left_lh[0] > cur_lh[0] and left_lh[1] > cur_lh[1]:
            left_arrow = 2
            last_lh = left_lh
            continue
        if left_lh[0] < cur_lh[0] and left_lh[1] < cur_lh[1]:
            left_arrow = 1
            last_lh = left_lh
            continue

    last_lh = None
    for right_step in range(1,5):
        if not left_chk:
            break
        if right_step+i >= quotes.shape[0]:
            break
        right_lh = (quotes.loc[i+right_step].low,quotes.loc[i+right_step].high)
        if right_arrow != -1:
            if right_lh[0] >= last_lh[0] and right_lh[1] <= last_lh[1]:
                continue

            if right_lh[0] > last_lh[0] and right_lh[1] > last_lh[1]:
                if right_arrow == 2:
                    right_chk = True
                    break
                if right_arrow == 1:
                    break

            if right_lh[0] < last_lh[0] and right_lh[1] < last_lh[1]:
                if right_arrow == 1:
                    right_chk = True
                    break
                if right_arrow == 2:
                    break

        if right_lh[0] >= cur_lh[0] and right_lh[1] <= cur_lh[1]:
            continue
        if right_lh[0] < cur_lh[0] and right_lh[1] > cur_lh[1]:
            break  # 右边边整条大于中间，直接退出
        if right_lh[0] > cur_lh[0] and right_lh[1] > cur_lh[1]:
            right_arrow = 2
            last_lh = right_lh
            continue
        if right_lh[0] < cur_lh[0] and right_lh[1] < cur_lh[1]:
            right_arrow = 1
            last_lh = right_lh
            continue

    print(i, left_arrow, right_arrow)
    if left_chk and right_chk and left_arrow==right_arrow:
        print("########",i, left_arrow)
        if left_arrow == 1:
            up_marks.append(True);down_marks.append(False)
        if left_arrow == 2:
            up_marks.append(False);down_marks.append(True)
    else:
        up_marks.append(False);down_marks.append(False)