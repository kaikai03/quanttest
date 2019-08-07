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

data = QA.QA_fetch_stock_min_adv("601155", '2019-8-06 9:00:00', '2019-8-06 15:00:00',frequence='5min').to_qfq().reset_index()
# data.close.plot()


df=pd.DataFrame(np.array([data.datetime,data.close,data.volume]).T,columns=['date','price','volume'])


# rearrange index
df.index=range(len(df.volume))

# check the total volume
print(np.sum(df.volume))

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
rolling_wondow = 2
N=50
# size=len(df.volume)  # volume size
size=np.sum(df.volume)/N  # volume size
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

# last_index.append(df.volume.index[-1])
# volume_bar_size.append(np.nansum(df.volume[last_index[-2]+1:last_index[-1]+1]))
last_price=df.price[last_index]

# test if volume bar is correct
print (np.sum(df.volume))
print (np.sum(volume_bar_size))

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
# d_price_diff_percentage.index=range(0,len(d_price_diff_percentage))
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

#sorted_vpin=np.sort(vpin_selected)
#yvals=1.*np.arange(len(sorted_vpin))/(len(sorted_vpin)-1)
#index=sorted(range(len(vpin_selected)),key=lambda x:vpin_selected[x])
#cdf_vpin=vpin_selected
#for i in range (0,len(vpin_selected)):
#    cdf_vpin[index[i-1]]=yvals[i-1]


# plot of VPIN vs Price
import matplotlib.pyplot as plt

# corresponded time index
time_bar_date=df.date[last_index]
b=[str(i) for i in time_bar_date]

# a=range(len(vpin))
a=[float(i) for i in last_index]


fig,ax1=plt.subplots(figsize=(13, 5))
ax1.plot(vpin,color='red',label='VPIN')
ax1.plot(cdf_vpin,'*-',color='brown')
ax1.set_ylabel('VPIN',color='red',fontsize=18)
ax1.set_xticks(a,minor=False)
ax1.set_xticklabels(b, rotation=60)
ax1.yaxis.grid()
ax1.xaxis.grid()
ax2=ax1.twinx()
ax2.plot(last_price[1:],color='blue',label='sell volume')
ax2.set_ylabel('Price',color='blue',fontsize=18)
ax2.plot(df.price,color='gray')


####################################################################

fig,ax1=plt.subplots()
ax1.plot(d_price_diff_percentage,color='red',label='VPIN')
ax1.set_ylabel('Price difference',color='red',fontsize=18)
ax1.set_xticks(a,minor=False)
ax1.set_xticklabels(b, rotation=60)
ax2=ax1.twinx()
ax2.plot(last_price,color='blue',label='sell volume')
ax2.set_ylabel('Price',color='blue',fontsize=18)

# find vpin of 5% worst price change bars
price_change=abs(d_price_diff_percentage)
import heapq as h
largest_deviation=h.nlargest(int(round(len(price_change)*0.05)),enumerate(price_change),key=lambda x:x[1])
largest_index,value=zip(*largest_deviation)
largest_vpin=[vpin[i] for i in largest_index]

pchange_vpin=np.zeros((len(largest_vpin),2))
pchange_vpin=pd.DataFrame(pchange_vpin,index=largest_index,columns=['price_deviation','vpin'])
pchange_vpin.price_deviation=value
pchange_vpin.vpin=largest_vpin


num=np.sum(vpin>min(largest_vpin))
den=len(vpin)
print (num/(den*1.0))

plt.figure()
plt.hist(vpin.dropna(),50)
aa=min(largest_vpin)
plt.plot((aa,aa),(0,10),lw=2)
plt.text(0.31,10,'Worst five percent price change')

print ("%.4f" % np.percentile(vpin,99))
print ("%.4f" % min(largest_vpin), "%.4f" % max(largest_vpin))

