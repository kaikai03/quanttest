# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:15:09 2015

@author: Terry
"""
import QUANTAXIS as QA
from QUANTAXIS.QAUtil.QAParameter import MARKET_TYPE

import pandas as pd
import numpy as np
import math
import scipy.stats as ss
import scipy.special as sp
import matplotlib.pyplot as plt
import datetime

pd.set_option('display.float_format', lambda x: '%.3f' % x)


def vpin_test(code, start, end, plot=False, frequence='5min',volumn_split=50,check_offset=12*2,rolling=2,cdf_vpin_threshold=0.8):
    print("start",code,start,end)
    try:
        data = QA.QA_fetch_stock_min_adv(code, start, end,frequence=frequence).to_qfq().reset_index()
    except:
        print("ERRORRRRR QA_fetch_stock_min_adv ERRORRRRR")
        return None
    # data.close.plot()


    df=pd.DataFrame(np.array([data.datetime,data.close,data.volume]).T,columns=['date','price','volume'])


    # rearrange index
    df.index=range(len(df.volume))


    # define function to group data accroding to volume size
    def volume(co,size):
        su=0
        while su<size:
              su=su+df.volume[co]
              co+=1
        return(su,co)


    # use the function to obtain last price and volume size of each volume bar
    counter=0
    count=0
    rolling_wondow = rolling
    N=volumn_split
    # size=len(df.volume)  # volume size
    size=np.nansum(df.volume)/N  # volume size
    last_index=[]
    volume_bar_size=[]
    try:
         while count<np.nansum(df.volume)/size:
                sum=volume(counter,size)[0]
                counter=volume(counter,size)[1]
                last_index.append(counter-1)
                volume_bar_size.append(sum)
                count+=1
    except:
          pass

    last_price=df.price[last_index]

    # test if volume bar is correct
    # print (np.sum(df.volume))
    # print (np.sum(volume_bar_size))

    price_volumebar=np.zeros((len(last_index),2))
    price_volumebar=pd.DataFrame(price_volumebar,index=last_index,columns=['price','bar_size'])
    price_volumebar.price=df.price[last_index]
    price_volumebar.bar_size=volume_bar_size

    # volume-weighted standard deviation
    def weighted_std(values, weights):
       mean = np.mean(values)
       weight_variance = weights*((values-mean)**2)
       weight_sum=np.sum(weights)
       variance_sum=np.sum(weight_variance)
       stDev=math.sqrt(variance_sum/weight_sum)
       return (stDev)


    d_price_diff=price_volumebar.price.diff()
    d_price_diff_percentage=d_price_diff/price_volumebar.price.shift(1)
    d_price_diff_percentage.index=range(0,len(d_price_diff_percentage))
    stDev=weighted_std(d_price_diff,price_volumebar.bar_size[1:])

    buy_sell=np.zeros((len(d_price_diff),4))
    buy_sell=pd.DataFrame(buy_sell,columns=['buy','sell','total','label'])

    # Applying BVC algorithm
    for i in range(0,len(d_price_diff)):
        if d_price_diff[last_index[i]] is np.nan:
            continue
        buy_sell.loc[i,'buy']=price_volumebar.bar_size[last_index[i]]*ss.t.cdf(d_price_diff[last_index[i]]/stDev,0.25)
        buy_sell.loc[i,'sell']=price_volumebar.bar_size[last_index[i]]-buy_sell.buy[i]
        buy_sell.loc[i,'total']=price_volumebar.bar_size[last_index[i]]



    # If more buys, then label the var as 'buy'
    for j in range(0,len(d_price_diff)):
        if buy_sell.buy[j]-buy_sell.sell[j]>0:
            buy_sell.loc[j,'label']='buy'
        else:
            buy_sell.loc[j,'label']='sell'

    buy_sell.index=last_index

    # caculate VPIN
    #buy_sell_vspread=buy_sell.sell-buy_sell.buy
    buy_sell_vspread=abs(buy_sell.sell-buy_sell.buy)
    rolling_size=rolling_wondow # rolling wondow size in terms of how many volume bars

    vpin=buy_sell_vspread.rolling(rolling_size).sum()/ buy_sell.total.rolling(rolling_size).sum()
    # rstd=last_price.rolling(rolling_size).std()


    ## calculate the cdf vpin
    def cdf(x):
        miu=np.nanmean(vpin)
        sigma=np.nanstd(vpin)
        cdf=0.5*(1+sp.erf((x-miu)/(2**0.5)/sigma))
        return (cdf)

    # cdf_vpin=[cdf(i) for i in vpin_selected]
    cdf_vpin = cdf(vpin)

    ###################################check#####################
    offset = check_offset
    checks = []
    checks_result = []

    for idx in cdf_vpin[cdf_vpin>cdf_vpin_threshold].index.values:
        size = len(df.price)
        if idx < size*0.05:continue

        if idx+offset > size-1:
            # dif = df.price[idx:size-1] - df.price[idx]
            dif = None
        else:
            dif = df.price[idx:idx+offset] - df.price[idx]

        if not dif is None:
            checks.append([idx, round(dif.max(), 4), round(dif.min(), 4), round(dif[-1:].values[0], 4),
                           round(abs(dif[-1:].values[0] / df.price[idx]), 4),
                           round(abs(dif.max() / df.price[idx]), 4),
                           round(abs(dif.min() / df.price[idx]), 4)
                           ])

    if len(checks)>0:
        checks_df = pd.DataFrame(checks, columns=['idx', 'max', 'min','last','lar', 'mar', 'mir'])
        checks_df = checks_df.set_index('idx')

        count_mar = round(checks_df[checks_df['mar']>0.02].shape[0]/checks_df.shape[0], 3)
        count_mir = round(checks_df[checks_df['mir']>0.02].shape[0]/checks_df.shape[0], 3)
        count_total = count_mar+count_mir

        print("total",checks_df.shape[0])
        print("2.0",count_mar,count_mir,count_total)
        checks_result.append([start, "2.0", checks_df.shape[0], count_mar,count_mir,count_total])

        count_mar = round(checks_df[checks_df['mar']>0.015].shape[0]/checks_df.shape[0], 3)
        count_mir = round(checks_df[checks_df['mir']>0.015].shape[0]/checks_df.shape[0], 3)
        count_total = count_mar+count_mir

        print("1.5",count_mar,count_mir,count_total)
        checks_result.append([start, "1.5", checks_df.shape[0], count_mar,count_mir,count_total])

        count_mar = round(checks_df[checks_df['mar']>0.01].shape[0]/checks_df.shape[0], 3)
        count_mir = round(checks_df[checks_df['mir']>0.01].shape[0]/checks_df.shape[0], 3)
        count_total = count_mar+count_mir

        print("1.0",count_mar,count_mir,count_total)
        checks_result.append([start, "1.0", checks_df.shape[0], count_mar,count_mir,count_total])
    else:
        print("checks hasn't exist")

    if plot:
        ##################################plot#####################
        # plot of VPIN vs Price
        # corresponded time index
        time_bar_date=df.date[last_index]
        b=[str(i) for i in time_bar_date]

        # a=range(len(vpin))
        a=[float(i) for i in last_index]


        fig,ax1=plt.subplots(figsize=(13, 5))
        # ax1.plot(vpin,color='red',label='VPIN')
        ax1.plot(d_price_diff,color='red',label='RT')



        ax1.plot(cdf_vpin,'*-',color='brown')
        ax1.set_ylabel('RT',color='red',fontsize=18)
        ax1.set_xticks(a,minor=False)
        # ax1.set_xticklabels(b, rotation=60)
        ax1.yaxis.grid()
        ax1.xaxis.grid()
        ax2=ax1.twinx()
        # ax2.plot(last_price[1:],color='blue',label='sell volume')
        ax2.plot(last_price[1:],color='blue',label='sell volume')
        ax2.set_ylabel('Price',color='blue',fontsize=18)
        ax2.plot(df.price,color='gray')
    return checks_result

def make_date_range(start_year_i=2016, end_year_i=2019,start_month=1):
    result = []
    for year in range(start_year_i,end_year_i,1):
        for month in range(1,12,1):
            for day in range(1,30,7):
                base = str(year)+"-"+str(month)
                start = base + "-" + str(day) + ' 9:00:00'
                if day+7 > 30:
                    break
                    # end = base + "-30" + ' 15:00:00'
                else:
                    end = base + "-" + str(day+7) + ' 15:00:00'
                result.append((start,end))
                if month==2 and day>=28:
                    break
    return result[start_month-1:]

date_ranges = make_date_range(start_month=10)

resualt = []
for date_range in date_ranges:
    tmp = vpin_test("600419", date_range[0], date_range[1], plot=False)
    if tmp != None:
        resualt.extend(tmp)

resualt_df=pd.DataFrame(resualt,columns=['date','type','count','mar','mir','r_total'])
resualt_df['date'] = pd.to_datetime(resualt_df['date'], format='%Y-%m-%d %H:%M:%S')
resualt_df['year'] =resualt_df['date'].apply(lambda x:datetime.datetime.strftime(x,'%Y'))


pd.to_datetime('2016-3-16 15:00:00', format='%Y-%m-%d %H:%M:%S')

plt.hist(resualt_df[resualt_df['type']=='2.0']['r_total'])
plt.hist(resualt_df[resualt_df['type']=='1.5']['r_total'])
plt.hist(resualt_df[resualt_df['type']=='1.0']['r_total'])

plt.hist(resualt_df[resualt_df['type']=='2.0']['mar'])
plt.hist(resualt_df[resualt_df['type']=='1.5']['mar'])
plt.hist(resualt_df[resualt_df['type']=='1.0']['mar'])

plt.hist(resualt_df[(resualt_df['type']=='2.0') & (resualt_df['year']=='2018')]['r_total'])
plt.hist(resualt_df[(resualt_df['type']=='2.0') & (resualt_df['year']=='2017')]['r_total'])
plt.hist(resualt_df[(resualt_df['type']=='2.0') & (resualt_df['year']=='2016')]['r_total'])

plt.hist(resualt_df[(resualt_df['type']=='1.0') & (resualt_df['year']=='2018')]['r_total'])
plt.hist(resualt_df[(resualt_df['type']=='1.0') & (resualt_df['year']=='2017')]['r_total'])
plt.hist(resualt_df[(resualt_df['type']=='1.0') & (resualt_df['year']=='2016')]['r_total'])


plt.hist(resualt_df[resualt_df['type']=='1.0']['mar'])
plt.hist(resualt_df[resualt_df['type']=='1.0']['mir'])



vpin_test("601155", '2016-3-15 9:00:00', '2016-3-16 15:00:00', plot=True)

data = QA.QA_fetch_stock_min_adv("601155", '2016-3-15 9:00:00', '2016-3-16 15:00:00',frequence="5min").to_qfq().reset_index()


# csv_data = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\distribut2-5.csv")
# csv_data.set_index(['Height','sex']).stack().to_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\distribut2-5_deal.csv")
#
#
# data_02 = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\fff0-2.csv")
#
# data_25 = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\fff2-5.csv")
#
#
#
# plt.plot(data_02[data_02["sex"]==0].loc[:,['Length','P3','P15','P50','P85','P97'] ].set_index(['Length']))
# plt.plot(data_25[data_25["sex"]==0].loc[:,['Length','P3','P15','P50','P85','P97'] ].set_index(['Length']))
