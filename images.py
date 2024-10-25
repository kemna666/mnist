import os 
import requests
import re
import time

def image_set(save_path,word,epoch):
    q=0
    a=0
    while True:
        time.sleep(1)
        url="https://images.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=index&fr=&hs=0&xthttps=111110&sf=1&fmq=&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word={}&oq={}&rsp=-1".format(word,word)
        headers={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET4.0C; .NET4.0E; rv:11.0) like Gecko"}
        response=requests.get(url,headers)
        html=response.text
        urls=re.findall('"objURL":"(.*?)"',html)
        for url in urls:
            print(a)
            response = requests.get(url,headers=headers)
            image = response.content
            with open(os.path.join(save_path,"{}.jpg".format(a))) as f:
                f.write(image)
            a=a+1
        q=q+20
        if(q/20)>=int(epoch):
            break
if __name__=="__main__":
    save_path = "/kemna"
    word=input('输入你要的图片')
    epoch=input('下载几轮图片？')
    image_set(save_path,word,epoch)