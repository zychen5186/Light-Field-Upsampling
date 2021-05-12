# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:51:21 2021

@author: brian
"""

import requests
import json

url = "https://www.google.com.tw/maps/preview/review/listentitiesreviews?authuser=0&hl=en&gl=tw&pb=!1m2!1y3765758546651144975!2y6093113884180453713!2m2!1i10!2i10!3e1!4m5!3b1!4b1!5b1!6b1!7b1!5m2!1soX-aYLr4BOeYr7wPp8Sl8AE!7e81"
text = requests.get(url).text

pretext = ')]}\''
text = text.replace(pretext,'')
soup = json.loads(text)

conlist = soup[2]
for i in conlist:
    print("username: " + str(i[0][1]))
    print("time: " + str(i[1]))
    print("comment: "+ str(i[3]))