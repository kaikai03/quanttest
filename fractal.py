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

correlate_stocks_all = ["000008","000020","000032","000049","000058","000063","000066","000158","000333","000404","000410","000413","000555","000584","000586","000651","000662","000665","000671","000676","000687","000711","000723","000816","000836","000837","000901","000909","000918","000922","000925","000938","000948","000961","000977","000988","000997","001696","001979","002006","002008","002009","002011","002025","002026","002031","002045","002047","002052","002055","002058","002062","002065","002073","002090","002121","002131","002139","002147","002151","002152","002161","002175","002177","002178","002179","002184","002197","002209","002212","002214","002226","002230","002232","002236","002241","002248","002253","002268","002270","002277","002283","002288","002292","002296","002298","002308","002312","002316","002331","002334","002337","002338","002347","002348","002351","002362","002367","002368","002369","002373","002380","002383","002384","002396","002401","002402","002403","002405","002410","002414","002415","002421","002426","002439","002444","002452","002456","002472","002474","002475","002497","002502","002512","002518","002520","002526","002527","002528","002535","002547","002559","002577","002583","002587","002591","002599","002600","002609","002611","002614","002635","002642","002655","002657","002660","002681","002689","002698","002747","002767","002777","002782","002819","002835","002849","002855","002857","002861","002866","002869","002870","002877","002881","002888","002902","002903","002917","002922","002925","300002","300007","300012","300014","300018","300020","300023","300024","300025","300033","300035","300036","300044","300048","300050","300053","300058","300066","300070","300074","300075","300076","300078","300083","300093","300096","300097","300098","300099","300101","300105","300112","300114","300115","300117","300123","300124","300126","300131","300136","300150","300154","300155","300161","300165","300167","300168","300173","300177","300182","300184","300188","300193","300195","300201","300203","300207","300209","300211","300212","300213","300217","300222","300227","300231","300248","300249","300253","300256","300259","300270","300275","300276","300278","300279","300281","300286","300287","300293","300297","300300","300306","300307","300310","300316","300322","300324","300328","300338","300345","300348","300349","300353","300354","300358","300360","300367","300368","300370","300371","300382","300397","300400","300403","300407","300410","300415","300416","300417","300420","300430","300433","300445","300448","300449","300462","300466","300475","300479","300480","300486","300515","300516","300523","300525","300543","300546","300552","300553","300557","300567","300572","300580","300588","300602","300603","300605","300607","300613","300647","300648","300667","300672","300679","300686","300691","300701","300709","300720","300735","600037","600071","600074","600100","600112","600152","600166","600167","600172","600198","600225","600260","600271","600288","600289","600336","600340","600345","600366","600386","600410","600476","600478","600481","600487","600503","600520","600522","600556","600560","600565","600601","600602","600637","600651","600654","600662","600690","600699","600701","600710","600718","600728","600734","600745","600770","600775","600797","600804","600831","600835","600843","600845","600855","600869","600894","601138","601222","601231","601238","601318","601567","601608","601727","601777","603015","603018","603030","603100","603131","603160","603189","603203","603357","603380","603486","603516","603528","603556","603633","603636","603656","603660","603666","603680","603728","603776","603828","603869","603890","603901","603933","603960"]

def get_data(code, start, end, frequences):
    if frequences == 1:
        data = QA.QA_fetch_stock_day_adv(code, start, end).to_hfq()
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

x_simple = []
y_label=[]
for stock in correlate_stocks_all:
    print("start:",stock)
    quotes, ind_MACD = get_data(stock, '2017-11-01', '2019-01-30', 1)  # 由于需要MACD信号，需要向前推33天。
    # quotes["ret"] = (quotes.low-quotes.low.shift(1))/quotes.low.shift(1)
    # quotes["vol_chg"] = (quotes.volume-quotes.volume.shift(1))/quotes.volume.shift(1)
    up_marks, down_marks, up2_marks, down2_marks = make_marks(quotes)

    for index in range(len(up_marks)):
        if index < 33: continue
        if down_marks[index]:  # up_marks[index] or
            # print(index,1,quotes.date[index])
            tmp = []
            tag="close"
            if up_marks[index]:tag="high"
            if down_marks[index]:tag="low"
            tmp.extend(np.round(quotes[tag][index-2:index+2].values,2))
            # tmp.extend(ind_MACD[index-2:index+2].MACD.values)
            # tmp.extend(ind_MACD[index-2:index+2].MACD.values/ind_MACD[index-2:index+2].MACD.max())
            # tmp.append(np.round(ind_MACD[index-2:index+2].MACD.skew(),2))
            # tmp.append(np.round(quotes.volume[index-2:index+2].skew(),2))
            # tmp.extend(quotes["vol_chg"][index-2:index+2].values)
            x_simple.append(tmp)
            y_label.append(1)

        if (down2_marks[index] and not down_marks[index]):  #(up2_marks[index] and not up_marks[index]) or
            # print(index, 0,quotes.date[index])
            tmp = []
            tag="close"
            if up2_marks[index]:tag="high"
            if down2_marks[index]:tag="low"
            tmp.extend(np.round(quotes[tag][index - 2:index + 2].values,2))
            # tmp.extend(ind_MACD[index-2:index+2].MACD.values)
            # tmp.extend(ind_MACD[index-2:index+2].MACD.values/ind_MACD[index-2:index+2].MACD.max())
            # tmp.append(np.round(ind_MACD[index-2:index+2].MACD.skew(),2))
            # tmp.append(np.round(quotes.volume[index-2:index+2].skew(),2))
            # tmp.extend(quotes["vol_chg"][index-2:index+2].values)
            x_simple.append(tmp)
            y_label.append(0)

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

fig = plt.figure("--", figsize=(16, 8))
ax1 = fig.add_subplot(111)
red_count = 0
for i,y in enumerate(y_label):
    x_np = np.array(x_simple[i])
    x_np = x_np/x_np.max()
    if y == 0:
        ax1.scatter([1,2,3,4], x_np, marker="_", c="blue")
    else:
        ax1.scatter([1.5,2.5,3.5,4.5], x_np, marker="_", c="red")
        red_count += 1

    if red_count==500:break

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
