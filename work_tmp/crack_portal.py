import numpy as np
f = open('./file/database_PORTAL.set','rb')

origin = f.readline()

len(origin)

origin[3]
ascii(origin[3])

def of_binary(decimal):
    al_decimal = decimal
    if al_decimal == 0 :
	    return "0"

    ls_binary = ""
    while(al_decimal >= 1):
        li_remainder = int(np.mod(al_decimal,2))
        al_decimal = al_decimal / 2
        ls_binary = str(li_remainder) + ls_binary

    return ls_binary

def of_bitwisenot_string(as_bit):
    li_cnt = len(as_bit)
    ls_result=""
    for i in range(0,li_cnt,1):

        if as_bit[i] == "0":
            ls_result = ls_result + "1"
        else:
            ls_result = ls_result + "0"
    return ls_result

def of_decimal(as_binary):
    ll_len = len(as_binary)
    ll_decimal = 0
    for i in range(0,ll_len,1):
        if (( not as_binary[i] == "1") and ( not as_binary[i] == "0")):
            return -1
        ll_decimal = ll_decimal + int(as_binary[i]) * 2 ** (ll_len-1 - i)

jump = False
ls_return = ""
for i in range(len(origin)):
    print(i, ":")

    if jump:
        jump=False
        continue

    ll_long = origin[i]
    print("ll_long:",ll_long)

    if ll_long > 128:
        i+=1
        jump=True

    ls_bit = of_binary(ll_long)

    ll_len = 8 - len(ls_bit)
    print("ll_len:",ll_len)

    for j in range(0,ll_len,1):
        ls_bit = "0" + ls_bit
    print("ls_bit:",ls_bit)

    ls_bit = of_bitwisenot_string(ls_bit)
    ll_code = of_decimal(ls_bit)
    ls_char = chr(ll_code)
    print("ls_char:",ls_char)
    ls_return = ls_return + ls_char
print(ls_return)



