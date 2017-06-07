from __future__ import print_function
#from urllib.request import urlopen # Python 3
from urllib import urlopen # Python 2
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

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
    
def get_bs_from_str(s):
    bs = BeautifulSoup(s)
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
    

def find_links(l1):
    links = []
    
    pp = l1.findAll("article-item",{"item":"property"})
    # len(pp)
    for p in pp: 
        aa = p.findAll("a",{"class":"u-url"})
        assert len(aa)==1, "Should be only one u-url for a property!"
        href = aa[0].attrs['href']
        if href.startswith('/nl/detail'):
            href = 'http://immo.vlan.be' + href
        links.append(href)
        
    return links
        
def save_html(filename,url='http://immo.vlan.be/nl/detail/VAF75192?r=s_d8a709b803d80bd9a6351c182d255e463f84c080'):
    html = urlopen(url)
    html_data = html.read()
    fp = open(filename,'w')
    print(html_data,file=fp)
    fp.close()
    print('Saved %s => %s' % (url,filename))

def save_files(l1, dirname):
    links = find_links(l1)
    
    for nhouse,url in enumerate(links):
        try:
            save_html('%s/house_%d.html'%(dirname, nhouse+1), url)
        except IOError:
            print('IOError: Skipping %s/house_%d.html'%(dirname, nhouse+1))
            
        toss = np.random.randint(2)
        if toss == 1:
            import time
            print('zzz...')
            time.sleep(1)
    print('ZZZZZZZZ....')
    time.sleep(1)
    
def save_multiple_files(startn = 1, endn = 5):
    n = startn   
    while n <= endn:
        filename = 'lists/list%d.html'%(n)
        clist = get_bs(filename)
        directory = 'alle/list%d'%(n)
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_files(clist, 'alle/list%d'%(n))
        n += 1
    
# Save the lists
def go_to_page(ff, page_number=1):
    find = ff.find_element_by_link_text
    page = str(page_number)
    try:
        elem = find(page)
        if not elem:
            print('Can\'t find page:',page_number)
            return
        elem.click()
    except: # NoSuchElementException
        return
    
def save_list(ff, page_number=1, dirname='lists'):
    src = ff.page_source
    filename='%s/list%d.html'%(dirname,page_number)
    fp = open(filename,'w')
    print(src.encode('utf8'),file=fp)
    print('Saved browser source => %s'%filename)
    
def save_pages(ff, start_page=1, end_page=11):
    every=100
    for pg in range(start_page, end_page):
        print('Go to page',pg)
        go_to_page(ff, pg)
        print('ZZZ...')
        time.sleep(20)
        save_list(ff, pg)
        if pg%every==0:
            print('Take a long rest...')
            time.sleep(300)
            
        

            
            
        
    

    

    

    