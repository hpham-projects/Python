# -*- coding: utf-8 -*-
# %reset # clear the variables in Ipyton
# Create plots with Bokeh in Python
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
# Use output_notebook if you are using an IPython or Jupyter notebook
#from bokeh.io import output_notebook
#output_notebook()
import os

#import seaborn as sns                        # Statistical visualization library based on Matplotlib

os.chdir('..')

# Load and clean data
# Data about dress
def get_data():
    def convert(c):
        if type(c).__name__ == "unicode": return c
        return c.strftime("%d/%m/%Y")
    
    dressdata = pd.read_excel('data/DressSales.xlsx',sheetname=0, header=0)
    dressdata.ix[0:5,1:24] # first five rows
    
    dressdata = \
    dressdata.rename(columns=lambda c: 
            c if type(c).__name__ == "unicode" else c.strftime("%d/%m/%Y") 
        )
                    
    dressfeatures = pd.read_excel('data/AttributeDataSet.xlsx',sheetname=0,header=0)    
    return dressdata, dressfeatures
#dressdata, dressfeatures = get_data()

def firstcap():
    global dressfeatures
    
    dressfeatures = dressfeatures.rename(columns={"waiseline":"WaistLine"})
    dressfeatures = dressfeatures.rename(columns={"Pattern Type":"PatternType"})
    
    for y in dressfeatures.columns:     
        x = np.unique(dressfeatures[y].values)
        for i in x:
            dressfeatures[y][dressfeatures[y]==i]=str(i).title()
#firstcap()

def remove_nulls():
    """ Replace "Null" with np.nan """
    global dressfeatures
    
    for c in ['Decoration', 'FabricType', 'Material', 'PatternType', 'WaistLine']:
        dressfeatures[c][dressfeatures[c]=='Null'] = np.nan
#remove_nulls()

def fix_spelling():
    def rename(c, bad, good):
        dressfeatures[c][dressfeatures[c]==bad]=good
    
    c = 'FabricType'
    for bad,good in [
            ('Flannael', 'Flannel'),
            ('Knitting', 'Knitted'),
            ('Sattin', 'Satin'),
            ('Wollen', 'Woolen'),
               ]:
        rename(c, bad, good)
        
    c = 'Material'
    for bad,good in [
            ('Model', 'Modal'),
            ('Sill', 'Silk'),
               ]:
        rename(c, bad, good)
        
    c = 'NeckLine'
    for bad,good in [
            ('Mandarin-Collor', 'Mandarin-Collar'),
            ('Peterpan-Collor', 'Peterpan-Collar'),
            ('Sqare-Collor', 'Square-Collar'),
            ('Turndowncollor', 'Turndowncollar'),
               ]:
        rename(c, bad, good)
        
    c = 'PatternType'
    for bad,good in [
            ('Leapord', 'Leopard'),
               ]:
        rename(c, bad, good)
    
    c = 'Season'
    for bad,good in [
            ('Automn', 'Autumn'),
               ]:
        rename(c, bad, good)

    c = 'Size'
    for bad,good in [
            ('Small', 'S'),
            ('Xl', 'XL'),
               ]:
        rename(c, bad, good)
    
    c = 'SleeveLength'
    for bad,good in [
            ('Cap-Sleeves', 'Capsleeves'),
            ('Half',        'Halfsleeves'),
            ('Halfsleeve',  'Halfsleeves'),
            ('Sleeevless',  'Sleeveless'),
            ('Sleevless',   'Sleeveless'),
            ('Sleveless',   'Sleeveless'),
            ('Threequater', 'Threequarter'),
            ('Thressqatar', 'Threequarter'),
            ('Turndowncollor',  'Turndowncollar'),
            ('Urndowncollor',   'Turndowncollar'),
               ]:
        rename(c, bad, good)   
#fix_spelling()

            
# Exploring the data
# ----------------

def read_cols(data):
    cols = {i for i in data.columns}
    return cols

# Make plots and important statistics
# Plot the 5 products with the best sales
dressdata.shape
dressdata['sumsales']=dressdata.ix[:,1:-1].sum(axis=1)
x = dressdata['sumsales'].sort_values()
best5 = dressdata.ix[x[-5:].keys(),:]
worst5 = dressdata.ix[x[0:5].keys(),:]

from bokeh.models import SingleIntervalTicker, LinearAxis
from bokeh.core.properties import Dict, Int, String

def get_lists():
    best5_1=best5.ix[best5.index[0],:]
    xis = np.arange(1,len(best5_1)+1)
    bb = [best5.ix[n,1:-1].values for n in best5.index]

    # Deal with nan values
    B = np.array(bb).astype(np.double)
    mask = np.isfinite(B)

    list_x = []
    list_y = []
    for i in np.arange(5):
        list_x.append(xis[mask[i]])
    for i in np.arange(5):
        list_y.append(bb[i][mask[i]])
    return list_x, list_y
        
def show_p2():
    p2 = figure(title='p3: Sales of 5 best selling products',
               x_axis_label='date',
               y_axis_label='daily sales')
    output_file('p3.html')
    lx_orig,ly = get_lists()
    lx = lx_orig
    
    colors_list=['blue','yellow','green','red','purple']
    legends_list = []
    labels = best5.ix[:,0]
    labels1 = ['ID '+str(labels[n]) for n in best5.index]
    
    for i in np.arange(5):
        legends_list.append(labels1[i])
    
    
    for (c,l,x,y) in zip(colors_list, legends_list, lx, ly):
        p2.line(x,y,color=c, legend=l, line_width=3)
        
    from bokeh.models import FuncTickFormatter
    p2.legend.location = 'bottom_right'
    
    # Change the xaxis label
    kk = best5_1.ix[1:-1].keys()
    kk_noyear = [d[0:-5] for d in kk] # Remove years (e.g. '/2013')
    kk_js = [str(s) for s in kk_noyear] # Remove u prefix (e.g. u'...') from strings.

    p2.xaxis[0].formatter = FuncTickFormatter(code="""
        var tlabels = %s;
        return tlabels[tick];
"""%repr(kk_js))    
    show(p2)
#show_p2()    

# Plot the style: pie-chart
# Group 5 main style and others
#dressfeatures['Style'][dressfeatures['Style']=='sexy']='Sexy'
# Make first letter capital
from bokeh.charts import Donut
def plot5pop():
    x = np.unique(dressfeatures['Style'].values)
    for i in x:
        dressfeatures['Style'][dressfeatures['Style']==i]=str(i).title()
    # print out the styles
    x = np.unique(dressfeatures['Style'].values)
    print('Styles are:')
    for i in x:
        print i
    # 
    grouped_style = dressfeatures.groupby(['Style'])
    cnt_grouped_style = grouped_style['Dress_ID'].count()
    cnt_grouped_style.sort_values(ascending=False, inplace=True)
    
    # Pick 5 most popular styles and combine the others as others
    list5style_labels = ['Casual', 'Sexy', 'Party', 'Cute', 'Vintage',
                  'Others']
    valuepopular5 = cnt_grouped_style[:5]
    valuepopular5['Others']=cnt_grouped_style.values[6:].sum()
    data = pd.Series(valuepopular5.values,index=list5style_labels)
    p = figure()
    pie_chart = Donut(data,title='Popular Styles')
    output_file("donut.html", title='Popular Styles')
    show(pie_chart)
plot5pop()


