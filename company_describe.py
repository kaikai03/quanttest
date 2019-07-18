import QUANTAXIS as QA
from QUANTAXIS.QAData.financial_mean import financial_dict
import datetime
import pandas as pd
import time

import matplotlib.pyplot as plt

# from scipy import stats


codes = QA.QA_fetch_stock_block_adv().get_block('人工智能').code
for i in QA.QA_fetch_stock_block_adv().block_name:print(i)

infos = QA.QA_fetch_stock_list_adv()

[(code, infos[infos.code==code].name[0]) for code in codes]

infos[infos.code==code]

code = '300558'
start_year = '2000'
end_year = '2019'

day = '2018-06-30'


Q4_list = [str(y)+'-12-31' for y in range(int(start_year), int(end_year))]


res_adv = QA.QA_fetch_financial_report_adv(code, Q4_list)
# res_adv.data.stack()

ROE = res_adv.get_key(code, [start_year,end_year], 'ROE')  # netProfit_rate*turnoverRatioOfTotalAssets*equityMultiplier
net_profit = res_adv.get_key(code, [start_year,end_year], 'netProfitFromOperatingActivities')  # 净利润
operating_revenue = res_adv.get_key(code, [start_year,end_year], 'operatingRevenue')  # 营业收入
net_profit_rate = net_profit/operating_revenue * 100

total_assets = res_adv.get_key(code, [start_year,end_year], 'totalAssets')  # 总资产
turnover_ratio_of_total_assets = operating_revenue/total_assets * 100 #总资产周转率

total_owners_equity = res_adv.get_key(code, [start_year,end_year], 'totalOwnersEquity')  # 所有者权益
equity_multiplier = res_adv.get_key(code, [start_year,end_year], 'equityMultiplier')  # 权益乘数 totalAssets/totalOwnersEquity


report = res_adv.get_key(code, [start_year,end_year], 'currentRatio')
report = res_adv.get_key(code, [start_year,end_year], 'acidTestRatio')
report = res_adv.get_key(code, [start_year,end_year], 'cashRatio')

net_profit_rate[:,code].plot(label='net_profit_rate')
turnover_ratio_of_total_assets[:,code].plot(label='turnover_ratio')
equity_multiplier[:,code].plot(label='equity_multiplier')
for i,j in list(zip(equity_multiplier[:,code].index.values,equity_multiplier.values)):
    plt.text(i,j,round(j,1))
ROE[:,code].plot(label='ROE')
plt.legend(loc=0)

res_adv.get_report_by_date(code, day)

plt.figure(figsize=(14,6))
ax = plt.subplot(121)
net_profit_rate[:,code].plot(label='net_profit_rate',ax=ax)
for i,j in list(zip(net_profit_rate[:,code].index.values,net_profit_rate.values)):
    plt.text(i,j,round(j,1))
