import QUANTAXIS as QA
from QUANTAXIS.QAData.financial_mean import financial_dict
import datetime
import pandas as pd
import time

import matplotlib.pyplot as plt

# from scipy import stats

def show_value_tip(plot, series):
    for i,j in list(zip(series.index.values,series.values)):
        plot.text(i,j,round(j,1),va="bottom",ha="center")

def get_Q4_list(start, end):
    return [str(y)+'-12-31' for y in range(int(start), int(end)+1)]

codes = QA.QA_fetch_stock_block_adv().get_block(['婴童概念','乳业']).code
for i in QA.QA_fetch_stock_block_adv().block_name:print(i)

QA.QA_fetch_stock_block_adv().get_code('600419').data

codes = ['600419','603238','603398','600887','600597']
start_year = '2010'
end_year = '2019'



def show_ROE(codes, start_year, end_year):
    assert type(codes) == list
    plt.figure(figsize=(16,7))
    Q4_list = get_Q4_list(start_year, end_year)

    if len(codes) > 1:
        ax1 = plt.subplot(221)
        ax1.set_title('ROE')
        ax2 = plt.subplot(222)
        ax2.set_title('net_profit_rate')
        ax3 = plt.subplot(223)
        ax3.set_title('turnover_ratio_of_total_assets')
        ax4 = plt.subplot(224)
        ax4.set_title('equity_multiplier')

    for code in codes:
        res_adv = QA.QA_fetch_financial_report_adv(code, Q4_list)

        # netProfit_rate*turnoverRatioOfTotalAssets*equityMultiplier
        ROE = res_adv.get_key(code, [start_year,end_year], 'ROE')
        net_profit = res_adv.get_key(code, [start_year,end_year], 'netProfitFromOperatingActivities')  # 净利润
        operating_revenue = res_adv.get_key(code, [start_year,end_year], 'operatingRevenue')  # 营业收入
        net_profit_rate = net_profit/operating_revenue * 100

        # 总资产
        total_assets = res_adv.get_key(code, [start_year,end_year], 'totalAssets')
        #总资产周转率
        turnover_ratio_of_total_assets = operating_revenue/total_assets * 100
        # 所有者权益
        total_owners_equity = res_adv.get_key(code, [start_year,end_year], 'totalOwnersEquity')
        # 权益乘数 totalAssets/totalOwnersEquity
        equity_multiplier = res_adv.get_key(code, [start_year,end_year], 'equityMultiplier')

        if len(codes) > 1:
            ROE[:,code].plot(label=code, ax=ax1)
            net_profit_rate[:,code].plot(label=code, ax=ax2)
            turnover_ratio_of_total_assets[:,code].plot(label=code, ax=ax3)
            equity_multiplier[:,code].plot(label=code, ax=ax4)
        else:
            ROE[:,code].plot(label='ROE')
            net_profit_rate[:,code].plot(label='net_profit_rate')
            turnover_ratio_of_total_assets[:,code].plot(label='turnover_ratio')
            equity_multiplier[:,code].plot(label='equity_multiplier')
            show_value_tip(plt, equity_multiplier[:,code])
            plt.legend(loc=0)
            plt.title(code)

    if len(codes) > 1:
        plt.subplots_adjust(hspace=0.4)
        ax1.legend(loc=0)
        ax2.legend(loc=0)
        ax3.legend(loc=0)
        ax4.legend(loc=0)

show_ROE(codes, start_year, end_year)

def show_item_compare(codes, start_year, end_year, item="ROE"):
    assert type(codes) == list
    assert len(codes) > 1
    Q4_list = get_Q4_list(start_year, end_year)

    for code in codes:
        res_adv = QA.QA_fetch_financial_report_adv(code, Q4_list)
        valus = res_adv.get_key(code, [start_year,end_year], item)
        valus[:,code].plot(label=code)
    plt.legend(loc=0)
    plt.title(item)

show_item_compare(codes, start_year, end_year, item="ROE")

def get_descript(codes, start_year, end_year, Q4=False):
    if Q4:
        Q4_list = get_Q4_list(start_year, end_year)
        res_adv = QA.QA_fetch_financial_report_adv(codes, Q4_list)
    else:
        res_adv = QA.QA_fetch_financial_report_adv(codes, start_year,end_year)

    if res_adv.data is None:
        return None
    else:
        return res_adv.data.stack()

get_descript(['600419'],'2010','2019')

def get_company_name(codes):
    infos = QA.QA_fetch_stock_list_adv()
    return [(code, infos[infos.code==code].name[0]) for code in codes]

get_company_name(codes)