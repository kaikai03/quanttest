# -*- coding: utf-8 -*-
import numpy as np


array = [10, 9, 2, 5, 7, 3, 101, 18,2,3,4,5,6,7,8]
max_len = -1
step = 0
for i in range(len(array)):
    tmp = array[i];
    tmp_len = 1
    for j in range(i+1,len(array)):
        step+=1
        if array[j] >= tmp:
            tmp_len += 1
            tmp = array[j]
    max_len = max(max_len, tmp_len)
print(max_len)



