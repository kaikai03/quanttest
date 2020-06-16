import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 300)


csv_data = pd.read_csv("E:\\工作文档\\DRIMS\\data\\WHO\\头围（全）0-5 unstack前.csv", engine='python')
stack_data = csv_data.set_index(['Month','sex']).stack()
floor = stack_data.groupby(['Month','sex']).apply(lambda x:x.shift(1).fillna(0))
final = pd.DataFrame({'head_min':floor,'head_max':stack_data}).reset_index()
final.rename(columns={'level_2':'percentile'}, inplace=True)
final.index.name = 'id'

final.to_csv("E:\\工作文档\\DRIMS\\data\\WHO\\头围（全）0-5_deal.csv")





##############################################################################
csv_data = pd.read_csv("E:\\工作文档\\DRIMS\\data\\WHO\\身高别体重（全）unstack前2-5.csv", engine='python')
stack_data = csv_data.set_index(['Height','sex']).stack()
final = stack_data.reset_index()
final.rename(columns={'level_2':'percentile'}, inplace=True)
final.rename(columns={'Height':'height'}, inplace=True)
final.rename(columns={0:'weight'}, inplace=True)
final.index.name = 'id'

final.to_csv("E:\\工作文档\\DRIMS\\data\\WHO\\身高别体重（全）2-5_deal.csv")




#
#
data_02 = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\身高年龄.csv")
#
data = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\周数据\\年龄别身高_week_origin.csv")
#
#
#
# plt.plot(data[ (data["sex"]==0) & (data["Week"]>21)& (data["Week"]<51)].loc[:,['Week','3','10','50','90','97'] ].set_index(['Week']))
plt.plot(data[data["sex"]==0].loc[:,['Length','P3','P15','P50','P85','P97'] ].set_index(['Length']))



import os
import re

for root, dirs, files in os.walk("F:\\Cache\\qt\\35331963\\FileRecv\\20200608\\20200608"):
    for f in files:
        if 'htm' in f:
            with open(os.path.join(root, f),'r') as op:
                line = op.readlines()
                m = re.findall(r'<font size="2" >.*?</font>', line[0], re.I)
                for i, item in enumerate(m):
                    if '年' in item:
                        age = m[i + 1].replace('<font size="2" >', '').replace('</font>', '')
                        if int(age) > 20:
                            print(os.path.join(root, f))
                            print(age)
                        break

