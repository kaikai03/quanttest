
import matplotlib.pyplot as plt
import matplotlib
import QUANTAXIS as QA
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)


code = '002415'
start = '2018-01-01'
end = '2018-12-30'
frequences = [1,'60min','30min','15min','5min']


codes = QA.QA_fetch_stock_block_adv().get_block(['智能交通','智慧城市','安防服务','人脸识别','机器人概念','电子制造',
                                                 '电器仪表']).code


data =QA.QA_fetch_stock_day_adv(['002821','300340','300452','300482'],'2017-01-05','2017-12-25').to_hfq()

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

# 寻找相关性
data = QA.QA_fetch_stock_day_adv(code,start,end).to_hfq().price_chg
data.index = data.index.droplevel(level=1)

data = QA.EMA(data, 12)

for stock in codes_with_name:
    try:
        correlation_data = QA.QA_fetch_stock_day_adv(stock[0], start, end).to_hfq().price_chg
    except AttributeError as e:
        print("jump",stock)
        continue
    correlation_data.index = correlation_data.index.droplevel(level=1)
    correlation_data = QA.EMA(correlation_data, 12)
    combine = pd.DataFrame({'x': data, 'y': correlation_data}).dropna()

    print(stock[0], ",", stock[1], ",", np.corrcoef(combine.x, combine.y)[0][1])



