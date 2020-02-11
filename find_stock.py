
import matplotlib.pyplot as plt
import matplotlib
import QUANTAXIS as QA
import pandas as pd
import numpy as np
from collections import Counter

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


code = '002415'
start = '2018-01-01'
end = '2018-12-30'
frequences = [1,'60min','30min','15min','5min']


codes = QA.QA_fetch_stock_block_adv().get_block(['智能交通','智慧城市','安防服务','人脸识别','机器人概念','电子制造',
                                                 '电器仪表']).code


data =QA.QA_fetch_stock_day_adv(['002821','300340','300452','300482'],'2017-01-05','2017-03-25').to_hfq()

# 股票所处板块
blocks_by_code = QA.QA_fetch_stock_block_adv().get_code("603933").data

# 股票信息
infos = QA.QA_fetch_stock_list_adv()

# 获取板块对应的股票列表
codes_in_block = QA.QA_fetch_stock_block_adv().get_block(['智能交通','智慧城市','安防服务','人脸识别','机器人概念',
                                                          '电子制造', '电器仪表']).data.reset_index()
# 去除多余信息，去除重复，仅股票保留代码和名字
codes_in_block = codes_in_block[["code","blockname"]].groupby(['code']).count().index.values
codes_with_name = [(code, infos[infos.code==code].name[0]) for code in codes_in_block]

def get_cor(price_df, n=None, rolling=1, nan_threhold=0.1):
    prc_uns = price_df.unstack()
    lost_too_much = prc_uns.columns[prc_uns.isnull().sum() > prc_uns.shape[0]*nan_threhold]
    prc_uns.drop(lost_too_much, axis=1,inplace=True)
    prc_uns = prc_uns.dropna()

    cor_mat = np.corrcoef(QA.EMA(prc_uns, rolling).dropna(),rowvar=False)
    return pd.Series(cor_mat[0], index=prc_uns.columns)


infos = QA.QA_fetch_stock_list_adv()
codes = QA.QA_fetch_stock_block_adv().get_block(['医药']).code

QA.QA_fetch_stock_block_adv().get_block(['智能交通']).data

def remove_st_in_codes(codes, infos=None):
    ##去除ST项
    if infos is None:
        infos_ = QA.QA_fetch_stock_list_adv()
    else:
        infos_ = infos
    contain_st = infos_.loc[codes].name.str.contains('st',case=False,na=False)
    st_code_list = infos_.loc[codes].name[contain_st].index.to_list()
    list(map(codes.remove, st_code_list))


prc = QA.QA_fetch_stock_day_adv(codes,'2018-01-01','2018-03-30').to_hfq().price_chg
prc = QA.QA_fetch_stock_day_adv(['002821','300340','300452','300482'],
                               '2018-01-01','2018-03-30').to_hfq().price_chg

def get_min_cor_list(price_df, n=None, rolling=1, nan_threhold=0.1):
    #投票法，找出相对最小集合。
    prc_uns = price_df.unstack()
    lost_too_much = prc_uns.columns[prc_uns.isnull().sum() > prc_uns.shape[0]*nan_threhold]
    prc_uns.drop(lost_too_much, axis=1,inplace=True)
    prc_uns = prc_uns.dropna()


    cor_mat = np.corrcoef(QA.EMA(prc_uns, rolling).dropna(),rowvar=False)
    c = Counter([np.abs(cor).argmin() for cor in cor_mat])
    mini_cor_code_args = [i[0] for i in c.most_common(n)]
    return prc_uns.columns[mini_cor_code_args].to_list()


min_cor = get_min_cor_list(prc,5)


prc2 = QA.QA_fetch_stock_day_adv(min_cor,
                                 '2018-01-01','2018-03-30').to_hfq().price
prc_uns2 = prc2.unstack()
lost_too_much2 = prc_uns2.columns[prc_uns2.isnull().sum() > prc_uns2.shape[0]*0.1]
prc_uns2.drop(lost_too_much2, axis=1,inplace=True)
prc_uns2 = prc_uns2.dropna()
cor_mat2 = np.corrcoef(QA.EMA(prc_uns2, 1).dropna(),rowvar=False)

