from scipy.stats import norm
import math
import pandas as pd

pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.width', 300)


def LMS(M,L,S,Z):
    if L.all() != 0:
        return M*(1+L*S*Z)**(1/L)
    else:
        return M*math.exp(S*Z)



def zscore(x,M,L,S):
    if L.all() != 0:
        return (((x/M)**L)-1)/(L*S)
    else:
        return math.log((x/M),math.e)/S


# FENTON_LENGTH
# FENTON_WEIGHT
# FENTON_HEADC
# FENTON_BMI

percentile_list = [0.1, 1, 3, 5, 10, 15, 25, 50, 75, 85, 90, 95, 97, 99, 99.9]

import json
with open("./file/GCCurveDataJSON.txt", 'r') as f:
    js_obj = json.load(f)

js_str = json.dumps(js_obj['FENTON_LENGTH']['data']['female'])

df = pd.read_json(js_str,orient='records')
df["sex"]=1
df['Week']= (df['Agemos']*4.348214285714286).astype(int)

for p in percentile_list:
    df[str(p)]=LMS(df["M"],df["L"],df["S"],norm.ppf(p/100))
df.loc[df['Week']==21,"Week"]=22
df.loc[df['Week']==48,"Week"]=49
df.loc[df['Agemos']==df.iloc[26]['Agemos'],"Week"]=48

df.to_csv("./file/fenton_length_girl_full.csv")

del df["Agemos"]
del df["L"]
del df["M"]
del df["S"]

df.set_index(['Week','sex']).stack().to_csv("./file/fenton_length_girl_deal.csv")