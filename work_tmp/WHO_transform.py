import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 300)

csv_data = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\head_full.csv")
csv_data.set_index(['Week','sex']).stack().to_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\head_deal.csv")
#
#
data_02 = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\身高年龄.csv")
#
data = pd.read_csv("C:\\Users\\kai_k_000.ABA\\Desktop\\周数据\\年龄别身高_week_origin.csv")
#
#
#
plt.plot(data[ (data["sex"]==0) & (data["Week"]>21)& (data["Week"]<51)].loc[:,['Week','3','10','50','90','97'] ].set_index(['Week']))
# plt.plot(data_25[data_25["sex"]==0].loc[:,['Length','P3','P15','P50','P85','P97'] ].set_index(['Length']))


