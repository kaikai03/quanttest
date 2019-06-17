
import QUANTAXIS as QA
import datetime
import pandas as pd

import analysis_tools as at

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.float_format', None)
import matplotlib.pyplot as plt

from scipy import stats

from QUANTAXIS.QAData.financial_mean import financial_dict



codes = QA.QA_fetch_stock_block_adv().get_block('人工智能').code
for i in QA.QA_fetch_stock_block_adv().block_name:print(i)

infos = QA.QA_fetch_stock_list_adv()

[(code, infos[infos.code==code].name[0]) for code in codes]

infos[infos.code=='300561']

# res_adv=QA.QA_fetch_financial_report_adv('000001','2017-01','2017-06')
# res_adv=QA.QA_fetch_financial_report_adv(['000001'],['2017-03-31'])
# res_adv.data

# res_adv.get_report_by_date('002747','2018-03-31')
# res_adv.get_key(['000001'], ['2017-03','2017-09'], 'ROE')
# res_adv.get_key('000001', '2017-03-31', 'ROE')

effect_duration = [2,5,15,30,80]

def getRelavent(code, duration, rep_name='ROE'):

    res_adv=QA.QA_fetch_financial_report_adv(code, None)
    stock_date =QA.QA_fetch_stock_day_adv(code).to_hfq()

    return_rates = {}
    report_values = {}
    dates = {}
    resualt = {}
    for date_range in duration:
        return_rates[date_range] = []
        report_values[date_range] = []
        dates[date_range] = []

    for date in res_adv.data.index.levels[0]:
        for date_range in duration:
            data_afer_report = stock_date.select_time(
                date+datetime.timedelta(days=1),
                date+datetime.timedelta(days=date_range)
            )
            if data_afer_report.len == 0:
                continue

            report = res_adv.get_key(code, date, rep_name)
            if report == 0:
                continue

            return_rate = data_afer_report.price_chg
            r_sum = return_rate.sum()

            return_rates[date_range].append(r_sum*100)
            report_values[date_range].append(report)
            dates[date_range].append(date)

    for date_range in duration:
        spear = stats.spearmanr(return_rates[date_range], report_values[date_range])
        kendall = stats.kendalltau(return_rates[date_range], report_values[date_range])
        pearson = stats.pearsonr(return_rates[date_range], report_values[date_range])
        resualt['code']  = code
        resualt['spear'+str(date_range)] = {"r":spear[0], "p":spear[1]}
        resualt['kendall'+str(date_range)] = {"r":kendall[0], "p":kendall[1]}
        resualt['pearson'+str(date_range)] = {"r":pearson[0], "p":pearson[1]}

    return resualt



p = pd.DataFrame()


for c in codes[8:15]:
    resualt = getRelavent(c, effect_duration,'netProfit')
    p = p.append(pd.DataFrame(resualt))
    print('finish ',c)

p.transpose()
p.to_csv('./file/relevent.csv')


stock_date =QA.QA_fetch_stock_day_adv("002747").to_hfq()
stock_date.date
res_adv=QA.QA_fetch_financial_report_adv("002747", None)
report = res_adv.get_key("002747", ['2001-01','2019-03'], 'ROE')
report.index.levels[0]

at.drawplot_xy([list(stock_date.date), list(report.index.levels[0])], [list(stock_date.close/10), list(report)])
