# -*- coding: utf-8 -*-
# Name: fractal strategy

import QUANTAXIS as QA
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import dates, ticker
import datetime
import matplotlib.finance as mpf

import myfeatures as mf

# from collections import Counter

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 600)


code = '002415'
start = '2018-06-29'
end = '2018-07-03'
colors = {-1:"green",1:"red",0:"grey"}
frequences = [1,'60min','30min','15min','5min']

# correlate to 002415
correlate_stocks = ["000687","002008","002009","002011","002055","002139","002230","002236",
                    "002241","002334","002351","002362","002396","002497","002600","002614",
                    "002635","002660","002681","600260","600345","601138","603015","603100",
                    "603203","603660","603728","603869","603933"]


def get_data(code, start, end, frequences):
    if frequences == 1:
        data = QA.QA_fetch_stock_day_adv("000687", start, end).to_hfq()
    else:  # min
        data = QA.QA_fetch_stock_min_adv(code, start, end, frequences).to_hfq()

    ind_MACD = None
    if len(data) > 30:
        ind_MACD = data.add_func(mf.MACD_JCSC).fillna(0)

    data = data.reset_index()

    if frequences != 1:  # min
        data['date'] = data['datetime']

    data['order'] = range(data.shape[0])
    quotes = data[['order', 'open', 'high', 'low', 'close', 'date', 'volume']]
    quotes['date'] = pd.to_datetime(quotes.date).dt.strftime('%m-%d %H:%M')

    return quotes, ind_MACD


def make_marks(quotes_data):
    up_marks = []
    down_marks = []
    up2_marks = []
    down2_marks = []
    for i in range(quotes_data.shape[0]):
        if i < 5:
            up_marks.append(False);
            down_marks.append(False);
            up2_marks.append(False);
            down2_marks.append(False);
            continue

        cur_lh = (quotes_data.loc[i].low, quotes_data.loc[i].high)
        left_arrow = -1  # 1 up 2 down -1 break
        right_arrow = -1  # 1 up 2 down -1 break
        left_chk = False
        right_chk = False
        last_lh = None
        for left_step in range(1, 5):
            left_lh = (quotes_data.loc[i - left_step].low, quotes_data.loc[i - left_step].high)
            if left_arrow != -1:
                if left_lh[0] >= last_lh[0] and left_lh[1] <= last_lh[1]:
                    continue

                if left_lh[0] > last_lh[0] and left_lh[1] > last_lh[1]:
                    if left_arrow == 2:
                        left_chk = True
                        break
                    if left_arrow == 1:
                        break

                if left_lh[0] < last_lh[0] and left_lh[1] < last_lh[1]:
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
        for right_step in range(1, 5):
            if not left_chk:
                break
            if right_step + i >= quotes_data.shape[0]:
                break
            right_lh = (quotes_data.loc[i + right_step].low, quotes_data.loc[i + right_step].high)
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

        if left_chk and right_chk and left_arrow == right_arrow:
            if left_arrow == 1:
                up_marks.append(True);down_marks.append(False)
            if left_arrow == 2:
                up_marks.append(False);down_marks.append(True)
        else:
            up_marks.append(False);down_marks.append(False)

        if left_chk and left_arrow == right_arrow:
            if left_arrow == 1:
                up2_marks.append(True);down2_marks.append(False)
            if left_arrow == 2:
                up2_marks.append(False);down2_marks.append(True)
        else:
            up2_marks.append(False);down2_marks.append(False)

    return up_marks, down_marks, up2_marks, down2_marks


def draw_ax(index, fig, quotes, ind_MACD, up_marks, down_marks, up2_marks, down2_marks):
    def format_date(x, pos=None):
        if x < 0: return ''
        if x > len(quotes.date) - 1: return ">" + quotes.date[len(quotes.date) - 1]
        return quotes.date[int(x)]

    ax1 = fig.add_subplot(5, 1, 1 + index)
    mpf.candlestick_ohlc(ax1, quotes.values, width=0.8, colorup='r', colordown='g')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(int(len(quotes.date) / 8)))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

    ax1.scatter(quotes[down2_marks].order, quotes[down2_marks].low * 0.990, marker='v', c="pink")
    ax1.scatter(quotes[up2_marks].order, quotes[up2_marks].high * 1.009, marker='^', c="pink")

    ax1.scatter(quotes[down_marks].order, quotes[down_marks].low * 0.997, marker='v', c="black")
    ax1.scatter(quotes[up_marks].order, quotes[up_marks].high * 1.002, marker='^', c="black")

    plt.xticks(fontsize=7)

    # if type(ind_MACD) is pd.core.frame.DataFrame:
    #     ax2 = fig.add_subplot(10, 2, 2 + index * 4)
    #     ax2.bar(range(len(ind_MACD.MACD)),
    #             ind_MACD.MACD,
    #             color=[colors[i] for i in np.sign(ind_MACD.MACD)])
    #     ax2.plot(range(len(ind_MACD.DIFF)), ind_MACD.DIFF, lw=0.5)
    #     ax2.plot(range(len(ind_MACD.DEA)), ind_MACD.DEA, lw=0.5)
    #     ax2.set_xticks([])
    #
    # ax3 = fig.add_subplot(10, 2, 4 + index*4)
    # ax3.bar(range(len(quotes.volume)),
    #         quotes.volume,
    #         color=[colors[i] for i in np.sign(quotes.close - quotes.open)])
    # ax3.set_xticks([])


######绘总图
if 1:
    fig = plt.figure(start + "~" + end + "--", figsize=(16, 8))
    fig.subplots_adjust(left=0.03, bottom=0.1, top=0.99, right=0.99)
    for index,frequence in enumerate(frequences):
        start_time = start
        end_time = end
        if frequence == 1:
            start_time = datetime.datetime.strptime(start, '%Y-%m-%d')
            start_time = (start_time - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
            end_time = datetime.datetime.strptime(end, '%Y-%m-%d')
            end_time = (end_time + datetime.timedelta(days=30)).strftime('%Y-%m-%d')

        if frequence == '60min':
            start_time = datetime.datetime.strptime(start, '%Y-%m-%d')
            start_time = (start_time - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        if frequence == '30min':
            start_time = datetime.datetime.strptime(start, '%Y-%m-%d')
            start_time = (start_time - datetime.timedelta(days=4)).strftime('%Y-%m-%d')

        quotes, ind_MACD = get_data(code, start_time, end_time, frequence)
        up_marks, down_marks, up2_marks, down2_marks = make_marks(quotes)
        draw_ax(index, fig, quotes, ind_MACD, up_marks, down_marks, up2_marks, down2_marks)
        print("endddd",index)

        if index == 5:break


#######训练集采样

x2_simple = []
y2_label=[]
for stock in ["000687","002008","002009","002011","002055"]:
    print("start:",stock)
    quotes, ind_MACD = get_data(stock, '2019-02-01', '2019-09-30', 1)  # 由于需要MACD信号，需要向前推33天。
    # quotes["ret"] = (quotes.close-quotes.close.shift(1))/quotes.close.shift(1)
    # quotes["vol_chg"] = (quotes.volume-quotes.volume.shift(1))/quotes.volume.shift(1)
    up_marks, down_marks, up2_marks, down2_marks = make_marks(quotes)

    for index in range(len(up_marks)):
        if index < 33: continue
        if up_marks[index] or down_marks[index]:  # up_marks[index] or
            # print(index,1,quotes.date[index])
            tmp = []
            tag="close"
            if up_marks[index]:tag="high"
            if down_marks[index]:tag="low"
            tmp.extend(np.round(quotes[tag][index-2:index+2].values,2))
            tmp.extend(ind_MACD[index-2:index+2].MACD.values)
            # tmp.extend(quotes["vol_chg"][index-2:index+2].values)
            x2_simple.append(tmp)
            y2_label.append(1)

        if (up2_marks[index] and not up_marks[index]) or (down2_marks[index] and not down_marks[index]):  #(up2_marks[index] and not up_marks[index]) or
            # print(index, 0,quotes.date[index])
            tmp = []
            tag="close"
            if up2_marks[index]:tag="high"
            if down2_marks[index]:tag="low"
            tmp.extend(np.round(quotes[tag][index - 2:index + 2].values,2))
            tmp.extend(ind_MACD[index - 2:index + 2].MACD.values)
            # tmp.extend(quotes["vol_chg"][index - 2:index + 2].values)
            x2_simple.append(tmp)
            y2_label.append(0)

    break

#######训练集肉眼校验
fig = plt.figure("--", figsize=(16, 8))
ax1 = fig.add_subplot(111)
ax1.bar(quotes.order,
        quotes.high-quotes.low,
        bottom=quotes.low,
        color=[colors[i] for i in np.sign(quotes.close-quotes.open)])
ax1.xaxis.set_major_locator(ticker.MultipleLocator(int(len(quotes.date)/8)))
ax1.scatter(quotes[down2_marks].order, quotes[down2_marks].low * 0.990, marker='v', c="pink")
ax1.scatter(quotes[up2_marks].order, quotes[up2_marks].high * 1.009, marker='^', c="pink")

ax1.scatter(quotes[down_marks].order, quotes[down_marks].low * 0.997, marker='v', c="black")
ax1.scatter(quotes[up_marks].order, quotes[up_marks].high * 1.002, marker='^', c="black")


##############################################################################

def draw_full_fluctuate(figure, subplot, quotes_data):
    ax1 = figure.add_subplot(subplot)
    ax1.bar(quotes_data.order,
            quotes_data.high-quotes_data.low,
            bottom=quotes_data.low,
            color=[colors[i] for i in np.sign(quotes_data.close-quotes_data.open)])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(int(len(quotes_data.date)/8)))

    ax1.scatter(quotes[down2_marks].order, quotes[down2_marks].low * 0.990, marker='v', c="pink")
    ax1.scatter(quotes[up2_marks].order, quotes[up2_marks].high * 1.009, marker='^', c="pink")

    ax1.scatter(quotes[down_marks].order, quotes[down_marks].low * 0.997, marker='v', c="black")
    ax1.scatter(quotes[up_marks].order, quotes[up_marks].high * 1.002, marker='^', c="black")


############交互
def OnMouseMotion(event):
    if not event.xdata:
        return
    if event.xdata < 0 or event.xdata>len(quotes.high)-1:return
    print(event.xdata)
    # ax = plt.gca()
    lines1 = ax1.plot([event.xdata,event.xdata],[quotes.low.min(),quotes.high.max()])
    ax1.figure.canvas.draw()
    lines2 = ax2.plot([event.xdata, event.xdata], [ind_MACD.DIFF.min(), ind_MACD.DIFF.max()])
    ax2.figure.canvas.draw()
    #删除之前的线条，进行更新
    l = lines1.pop(0);l.remove();del l
    l = lines2.pop(0);l.remove();del l
fig.canvas.mpl_connect('motion_notify_event',OnMouseMotion)
############交互


#######横向对比及网格搜索
# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(random_state=1)
# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')  # 无权重投票
# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='soft', weights=[2,1,2]) # 权重投票
# for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
#     scores = cross_val_score(clf,X,y,cv=5, scoring='accuracy')
#     print("准确率: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#
# from sklearn.model_selection import GridSearchCV
# params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}  # 搜索寻找最优的lr模型中的C参数和rf模型中的n_estimators
# grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
# grid = grid.fit(iris.data, iris.target)
# print('最优参数：',grid.best_params_)
