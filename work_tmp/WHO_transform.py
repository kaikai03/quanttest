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


####################################odt文件##########################################
import os
import re
from odf import text, teletype
from odf.opendocument import load

result_list = []
for root, dirs, files in os.walk("F:\\Cache\\qt\\35331963\\FileRecv\\20200608\\20200608\\"):
    for f in files:
        # if 'htm' in f:
        #     with open(os.path.join(root, f),'r') as op:
        #         line = op.readlines()
        #         m = re.findall(r'<font size="2" >.*?</font>', line[0], re.I)
        #         for i, item in enumerate(m):
        #             if '年' in item:
        #                 age = m[i + 1].replace('<font size="2" >', '').replace('</font>', '')
        #                 if int(age) > 20:
        #                     print(os.path.join(root, f))
        #                     print(age)
        #                 break
        jump_blank_space_count = 0
        if 'odt' in f:
            textdoc = load(os.path.join(root, f))
            allparas = textdoc.getElementsByType(text.P)
            # teletype.extractText(allparas[0])
            # print('textdoc',textdoc)
            # print('allparas', allparas)
            # print('extract1', teletype.extractText(allparas[1]))
            # print('ext', teletype.extractText(allparas[1]),teletype.extractText(allparas[7]))

            for i,t in enumerate(allparas):
                content = teletype.extractText(t)
            #     print(i,'---',content)
            #     if '超声提示：' == content:
            #         jump_blank_space_count = 3
            #     if jump_blank_space_count != 0:
            #         if len(content) == 0:
            #             jump_blank_space_count -= 1
            #         else:
            #             print(teletype.extractText(allparas[1]),content)

                if '合并' in content:
                    print(teletype.extractText(allparas[1]), content)
                    result_list.append(teletype.extractText(allparas[1])+":"+content)
    print('---------------------')

print(set(result_list))




#################################a_test_z_report#############################################
import pandas as pd

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 300)

head = {
    "col_9": "卵圆孔大小",
    "col_10": "房间隔缺损部位",
    "col_11": "房间隔缺损大小",
    "col_12": "房间隔缺损类型",
    "col_13": "心房水平",
    "col_14": "房间隔缺损数量",
    "col_15": "三尖瓣反流压差",
    "col_16": "肺静脉回流情况",
    "col_17": "卵圆孔",
    "col_18": "房间隔",
    "col_19": "室间隔缺损部位",
    "col_20": "室间隔缺损大小",
    "col_21": "心室水平分流方向",
    "col_22": "室间隔缺损数量",
    "col_23": "室膈瘤有/无",
    "col_24": "室间隔分流口大小",
    "col_25": "室间隔分流速度",
    "col_26": "室间隔压差",
    "col_27": "动脉导管内径",
    "col_28": "大动脉水平分流",
    "col_29": "肺动脉端位置",
    "col_30": "动脉导管分流速度",
    "col_31": "室膈瘤有/无",
    "col_32": "动脉导管压差",
    "col_33": "肺动脉跨瓣压差",
    "col_34": "右心室肥大是/否",
    "col_35": "肺动脉瓣膜情况",
    "col_36": "肺动脉瓣环内径",
    "col_37": "肺动脉宽度",
    "col_38": "肺动脉血流速度",
    "col_39": "主动脉压差",
    "col_40": "主动脉血流速度",
    "col_41": "升主动脉内径",
    "col_42": "主动脉窦内径",
    "col_43": "主动脉瓣环内径",
    "col_44": "主动脉瓣叶情况",
    "col_45": "左心室肥大是/否",
    "col_47": "最窄处血流压差",
    "col_48": "最窄处血流速度",
    "col_49": "主动脉内径",
    "col_50": "狭窄部位",
    "col_52": "最窄处内径",
    "col_53": "狭窄段长度"
}

csv_data = pd.read_excel("C:\\Users\\fakeQ\\Desktop\\a_test_z_report.xlsx")
csv_data = pd.read_excel("C:\\Users\\fakeQ\\Desktop\\a_test_z_report_tmp.xlsx")
chaoDesc = csv_data["chaoDesc"]
del csv_data["chaoDesc"]
del csv_data["col_31"]
del csv_data["col_54"]
del csv_data["col_51"]
del csv_data["col_46"]
del csv_data["col_55"]
del csv_data["id"]
csv_data.rename(columns=head,inplace=True)

stack_data = csv_data.stack()
stack_data_full = csv_data.fillna("NULL").stack()
final = []
for index in range(len(stack_data.index.levels[0])-1):
    # print("\n\n\n",stack_data.loc[index])
    items = stack_data.loc[index]
    items_full = stack_data_full.loc[index]
    content, content_full = "",""
    for key in items.keys():
        content += key + " : " + items[key] + '\n'
    for key in items_full.keys():
        content_full += key + " : " + items_full[key]+'\n'
    print(index, content, '\n')
    final.append({'chaoDesc': chaoDesc[index], 'content': content, 'content_full': content_full})

pd.DataFrame(final).to_excel("C:\\Users\\fakeQ\\Desktop\\a_test_z_report_finish.xlsx")

