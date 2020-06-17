# coding=utf-8
import pandas as pd
import json
import numpy as np
import requests

pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)

demo = "心脏位置及连接正常。右房右室无明显增大。肺动脉血流1.1m/s，左肺动脉开口内径0.45cm，血流1.7m/s；右肺动脉开口内径0.35cm，血流1.8m/s。各瓣膜开放活动可，三尖瓣轻微反流。房间隔连续性中断约0.22cm（继发型），左向右分流。室间隔连续。左位主动脉弓。未见明显动脉导管开放征象。目前左右冠状动脉近段未见增宽。未见心包积液。组织多普勒测房室瓣E/A>1。"

def reqe(origin):
    # 发送post请求，带json串
    json_data = {"origin": origin}
    req = requests.post("http://180.76.56.111:8383/api/echoCG", json=json_data)
    js = req.json()
    if js is None:
        return None
    tmpStr = ''
    for item in json.loads(js):
        print(item)
        tmp = '(' + str(item['Key']) + '：' + str(item['Value']) + ')； '
        tmpStr += tmp
    return tmpStr.replace('None','')

def reqe_df(origin):
    # 发送post请求，带json串
    json_data = {"origin": origin}
    req = requests.post("http://180.76.56.111:8383/api/echoCG", json=json_data)
    js = req.json()
    if js is None:
        return None
    tmp = {}
    tmp["origin"] = origin
    for item in json.loads(js):
        print(item)
        tmp[str(item['Key'])] = str(item['Value']).replace('None','')

    return tmp

filename = 'C:\\Users\\fakeQ\\Desktop\\测试心超报告 .csv'


# 读取整个csv文件
csv_data = pd.read_csv(filename, engine='python')



for index, row in csv_data.iterrows():
    print(row['报告内容'])
    csv_data.iloc[index, 1] = reqe(row['报告内容'])


csv_data.to_excel("C:\\Users\\fakeQ\\Desktop\\结构化结果.xls")


#####################################
items = []
for index, row in csv_data.iterrows():
    print(row['报告内容'])

    items.append(reqe_df(row['报告内容']))

df = pd.DataFrame(items)
df.to_excel("C:\\Users\\fakeQ\\Desktop\\结构化结果_df.xls")