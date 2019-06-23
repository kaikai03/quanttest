# coding=utf-8
import pandas as pd
import json
import numpy as np
import requests

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)

demo = "2007/8/3 , 228905 , ** 病理号 : 2007-08922 病理检查时间 : 2007-08-03 病理诊断 : （ 胰腺 ） 导管腺癌 ， 中～低分化 ， 伴大片坏死 。 肿瘤大小3x2.3x2.2cm 。 胰腺切缘及十二指肠均未见癌累及 。 胰腺旁淋巴结1枚 ， 未见癌转移 。 慢性胆囊炎 。 \\@\\ 冰冻送检 （ 肝面结节 ） 为胶原结节 ， 0.5x0.5x0.1cm 。"

def reqe(origin):
    # 发送post请求，带json串
    json_data = {"origin": origin}
    req = requests.post("http://180.76.56.111:8383/api/med", json=json_data)
    js = req.json()
    if js is None:
        return None
    tmpStr = ''
    for item in json.loads(js):
        print(item)
        tmp = '(' + str(item['Key']) + '：' + str(item['Forward']) + str(item['Backward']) + ')； '
        tmpStr += tmp
    return tmpStr.replace('None','')

filename = '600例学习数据.csv'


# 读取整个csv文件
csv_data = pd.read_csv(filename, engine='python')


for index, row in csv_data.iterrows():
    print(row['病理报告'])
    csv_data.iloc[index, 3] = reqe(row['病理报告'])


csv_data.to_excel("结构化结果.xls")