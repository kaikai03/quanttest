# coding=utf-8
#
# The MIT License (MIT)
#
# Copyright (c) 2016-2017 yutiansut/QUANTAXIS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import  matplotlib
import QUANTAXIS as QA
from QUANTAXIS import QA_Backtest as QB
import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_rows', 5000)
import numpy as np


userA=QA.QA_User()

PortfolioA1=userA.new_portfolio()
PortfolioA2=userA.new_portfolio()

userA.portfolio_list
userA.user_cookie
userA.get_portfolio()

strategy1=PortfolioA1.new_account()
strategy2=PortfolioA1.new_account()


# 创建一个策略 自定义on_bar事件
class Strategy3(QA.QA_Strategy):
  def __init__(self,  user_cookie,portfolio_cookie):
    super().__init__(user_cookie,portfolio_cookie)

  def on_bar(self,event):
    print(event)

# 实例化该策略到strategy3
strategy3=Strategy3( userA.user_cookie, PortfolioA2.portfolio_cookie)

# 把该策略加载到A2组合中
PortfolioA2.add_account(strategy3)



