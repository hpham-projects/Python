from __future__ import print_function
#from urllib.request import urlopen # Python 3
from urllib import urlopen # Python 2
from bs4 import BeautifulSoup
import re
import pandas as pd

# load the website
#html = urlopen("samples/house1.htm")


#fp = open('samples/house1.htm')
#lines = fp.readlines()
#ss = '\n'.join(lines)
#bs = BeautifulSoup(ss)
import os
os.chdir('..')

def get_bs(filename):
    fp = open(filename)
    lines = fp.readlines()
    ss = '\n'.join(lines)
    bs = BeautifulSoup(ss)
    return bs
    
def find_detail(bs, detailText = "Aantal slaapkamers"):
    ul_int = bs.findAll("ul",{"class":"box"})
    for ul in ul_int:
        ul_details = ul
        for c in ul_details.children:
            if hasattr(c,'text') and   detailText in c.text:
                ss = c.findAll("span",{"class":"right"})
                #print "found the slaapkamers:",ss[0].get_text()
                return ss[0].get_text()

def find_price(bs):
    pp = bs.findAll("span", {"class":"p-price"})
    assert len(pp)==1, "There should be only one p-price on the page!"
    pp[0].get_text()
    price = pp[0].get_text()
    return price

def find_city(bs):
    cc = bs.findAll("li", {"class":"city"})
    assert len(cc)==1, "There should be only one city li on the page!"
    cc[0].get_text()
    city = cc[0].get_text()
    return city.strip()

interesting_details = [
    "Aantal slaapkamers",
    "Bewoonbare oppervlakte",
    "Aantal badkamers",
    "Aantal toiletten",
    "Oppervlakte terrein",
    "Bouwjaar",
    "Tuin",
    "Garage",
    "Kelder",
    "Aantal gevels",
]
    
def details(bs):
    for dtext in sorted(interesting_details):
        detail = find_detail(bs, dtext)
        if detail:
            print('%s: %s'%(dtext, detail))
            
def show_house(filename='samples/house1.htm'):
    bs = get_bs(filename)
    print('\n\n---------------\nFilename:',filename)
    c = find_city(bs)
    print('City:',c)
    p = find_price(bs)
    print('Price:',p)
    print('Details\n-------')
    details(bs)

def show_houses():
    for n in range(1,6):
        show_house('samples/house%d.htm'%n)

def json_houses():
    houses = []

    def short_name(long_name):
        #long_name = "Aantal slaapkamers"
        words = long_name.split(' ')
        last = words[-1]
        return last.lower()
        
    for n in range(1,6):
        filename = 'samples/house%d.htm'%n
        
        h = {}
        bs = get_bs(filename)
        h['__filename'] = filename 
        h['city'] = find_city(bs)
        h['price'] = find_price(bs)
    
        h['slaapkamers'] = None
        h['oppervlakte'] = None
        h['badkamers'] = None
        h['toiletten'] = None
        h['terrein'] = None
        h['bouwjaar'] = None
        h['tuin'] = None
        h['garage'] = None
        h['kelder'] = None
        h['gevels'] = None
        
        for dtext in sorted(interesting_details):
            detail = find_detail(bs, dtext)
            if detail:
                sn = short_name(dtext)
                h[sn] = detail

        houses.append(h)
    
    return houses
    
def df_houses():
    hdata = pd.DataFrame()
    hdata['__filename'] = ''
    hdata['city'] = ''
    hdata['price'] = ''
    hdata['slaapkamers'] = ''
    hdata['oppervlakte'] = ''
    hdata['badkamers'] = ''
    hdata['toiletten'] = ''
    hdata['terrein'] = ''
    hdata['bouwjaar'] = ''
    hdata['tuin'] = ''
    hdata['garage'] = ''
    hdata['kelder'] = ''
    hdata['gevels'] = ''

    def short_name(long_name):
        #long_name = "Aantal slaapkamers"
        words = long_name.split(' ')
        last = words[-1]
        return last.lower()
        
    for n in range(1,6):
        filename = 'samples/house%d.htm'%n
        
        h = {}
        bs = get_bs(filename)
        h['__filename'] = filename 
        h['city'] = find_city(bs)
        h['price'] = find_price(bs)
    
        h['slaapkamers'] = None
        h['oppervlakte'] = None
        h['badkamers'] = None
        h['toiletten'] = None
        h['terrein'] = None
        h['bouwjaar'] = None
        h['tuin'] = None
        h['garage'] = None
        h['kelder'] = None
        h['gevels'] = None
        
        for dtext in sorted(interesting_details):
            detail = find_detail(bs, dtext)
            if detail:
                sn = short_name(dtext)
                h[sn] = detail

        hdata = hdata.append( 
            pd.DataFrame([
                    [
                        h['__filename'],
                        h['city'],
                        h['price'],
                        h['slaapkamers'],
                        h['oppervlakte'] ,
                        h['badkamers'] ,
                        h['toiletten'] ,
                        h['terrein'] ,
                        h['bouwjaar'] ,
                        h['tuin'] ,
                        h['garage'] ,
                        h['kelder'] ,
                        h['gevels'] ,
                    ]
                ], columns=[
                        '__filename',
                        'city',
                        'price',
                        'slaapkamers',
                        'oppervlakte' ,
                        'badkamers' ,
                        'toiletten' ,
                        'terrein' ,
                        'bouwjaar' ,
                        'tuin' ,
                        'garage' ,
                        'kelder' ,
                        'gevels' ,
                ]) 
        )
    
    return hdata
    
def write_csv(filename='hdata.csv'):
    hdata = df_houses()
    hdata.to_csv(filename,encoding='utf8')
    print('CSV written to:',filename)
    
    
write_csv()

    

    

    