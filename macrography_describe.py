import QUANTAXIS as QA
import random
import time
import numpy as np
import pandas as pd
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 500)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

from multiprocessing.dummy import Pool

class myError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def count_continuous(sign_Series):
    max_continuous_positive_count=0
    max_continuous_negative_count=0
    continuous_tem = 0
    for idx,item in enumerate(sign_Series):
        if idx==0:continue
        if item==0:continue
        if item != sign_Series.iloc[idx-1]:
            continuous_tem=0
            continue

        continuous_tem+=1
        if item > 0:
            max_continuous_positive_count = max(max_continuous_positive_count,continuous_tem)
        else:
            max_continuous_negative_count = max(max_continuous_negative_count,continuous_tem)

    return max_continuous_positive_count,max_continuous_negative_count

def cross_describe(codes, start='2010-01-01',end='2019-06-30', bench=False, MA=0):
    #code = "000001",start='2015-01-01',end='2018-12-30'
    st_time = time.time()

    items = []

    def deal(code):
        print(" process:",code)
        stock_name = None

        if not str(code)[0] in ['0','6','b']:
            print("jump:", code)
            return

        if code == "bench":
            data = QA.QA_fetch_index_day_adv('000001',start,end)
            stock_name = "bench"
        else:
            data = QA.QA_fetch_stock_day_adv(code,start,end)


        if data is None or len(data)==0:
            print("jump:", code)
            return


        if code in code_in_infos:
            stock_name = stock_infos.loc[code]['name']

        if code != "bench":
            data = data.to_qfq()


        years_list = data.date.year.value_counts()
        # year = years_list.index[0]

        tmp = []
        for year in years_list.index:
            open_days = years_list[year]
            period_data = data.select_time(str(year),str(year+1))
            if len(period_data) == 0:
                continue

            if MA > 0:
                period_data = period_data.add_func(QA.MA, MA)
                period_data.dropna(inplace=True)

            priod_pct = period_data.close/period_data.close.shift(1) - 1
            priod_pct_shift = priod_pct.shift(1)
            sig = np.sign(priod_pct)
            up_down_count = sig.value_counts()

            compare = np.sign(priod_pct) == np.sign(priod_pct_shift)

            cross = compare.value_counts().get(False,0)
            one_day_con_rate = round(compare[sig>0].value_counts().get(True,0)/open_days,2)

            co_pos,cp_neg = count_continuous(sig)

            # "days":open_days,  "cross":cross,
            item={"code":code, "name":stock_name, "year":year,
             "co_p":co_pos, "co_n":cp_neg,
             "cros_r":round(cross/open_days,2),"cros1_r":one_day_con_rate,
             "go_up":up_down_count.get(1,None),"go_dn":up_down_count.get(-1,None),
             "max_pri":round(period_data.close.max(),2),"min_pri":round(period_data.close.min(),2)}

            tmp.append(item)
        items.extend(tmp)

    stock_infos = QA.QA_fetch_stock_list_adv()
    code_in_infos = stock_infos.code
    if bench:
        codes.insert(0,'bench')

    pool = Pool(4)
    pool.map(deal, codes)
    pool.close()
    pool.join()


    # for idx, code in enumerate(codes):


    df = pd.DataFrame(items)
    df.set_index(["code","name","year"], inplace=True)
    print("used:",(time.time()-st_time))
    return df


#######################test
codes = QA.QA_fetch_stock_block_adv()

len(codes.code)
df = cross_describe(codes.code, bench=True, MA=5)
df
df.sort_index().to_excel('./file/cross_describe_ma5.xls')
#######################test