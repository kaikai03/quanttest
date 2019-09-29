from selenium import webdriver
import time
import requests
b = webdriver.Firefox()
#b = webdriver.Ie()
b.delete_all_cookies()
cookies = {}
while True:
    list_cookies = b.get_cookies()
    print(list_cookies)
    for s in list_cookies:
        cookies[s['name']] = s['value']
    print(cookies)
    if cookies.get('JSESSIONID',None):
        # b.close()
        break
    time.sleep(10)

sn = requests.Session()
requests.utils.add_dict_to_cookiejar(sn.cookies, cookies)

url_base = 'http://xxx.xxx.cn/resource/'
extent_list = ['v/reports/2019-07-23/398115f482e34cbd97cc8df5b0234013.jpg']


def get_img(extent):
    response = sn.get(url = url_base+extent)
    img = response.content
    file_name = extent.split('/')[-1]
    path = './file/'+ file_name
    with open( path,'wb' ) as f:
        f.write(img)


list(map(get_img,extent_list))

