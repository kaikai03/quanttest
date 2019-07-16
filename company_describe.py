import QUANTAXIS as QA
from QUANTAXIS.QAData.financial_mean import financial_dict
import datetime
import pandas as pd

# import matplotlib.pyplot as plt

# from scipy import stats

codes = QA.QA_fetch_stock_block_adv().get_block('人工智能').code
for i in QA.QA_fetch_stock_block_adv().block_name:print(i)

infos = QA.QA_fetch_stock_list_adv()

[(code, infos[infos.code==code].name[0]) for code in codes]

infos[infos.code=='300561']

res_adv=QA.QA_fetch_financial_report_adv('002415','2010-01','2019-08')
res_adv.data