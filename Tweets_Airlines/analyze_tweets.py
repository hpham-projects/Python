import numpy as np
import pandas as pd
import matplotlib as plt
from pylab import *
from __future__ import division

# Date set: Twitter Airline Sentiments
# ------------------------------------

def readcols(data):
    for i in sorted(data.columns):
        print i
        
def tidydata():
    # load the data set: tweet_id, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence,
    # airline, airline_sentiment_gold, name, negativereason_gold, retweet_count, text, tweet_coord, tweet_created, tweet_location, 
    # user_timezone
    data = pd.read_csv('Tweets.csv')
    
def groupfre(dataFrame):
    arr1 = dataFrame.values
    arr2 = dataFrame.sum(axis=1) # sum across columns
    arr2 = [arr2]*3
    arr2 = np.transpose(arr2)
    
    arr1.astype(float)
    arr2.astype(float)
    return perct = arr1/arr2

def exploredata()
    # count sentiments by airlines
    al_grouped = data.groupby(['airline','airline_sentiment'])
    al_grouped['airline_sentiment'].count()
    alsent_cnt = al_grouped['airline_sentiment'].count()
    alsent_cnt = alsent_cnt.unstack()
    # number of tweets for all airlines & pie plot
    alsent_cnt_sum = alsent_cnt.sum(axis=1)
    airlines_names = alsent_cnt_sum.index
    pie(alsent_cnt_sum.values,labels=airlines_names,autopct='%1.1f%%')
    title('Airlines: number of tweets', bbox={'facecolor':'0.8', 'pad':2})
    show()
    # types of tweets for each airline & bar plot
    # in absolute values
    ind = np.arange(6) # locations for groups
    width = 0.35 # width of bars
    p1 = plt.bar(ind,alsent_cnt['negative'],color='r')
    p2 = plt.bar(ind,alsent_cnt['neutral'],color='k')
    p3 = plt.bar(ind,alsent_cnt['positive'],color='g')
    
    plt.xticks(ind,('American','Delta','Southwest','US', 'United', 'Virgin'))
    plt.legend((p1[0], p2[0], p3[0]),('Negative','Neutral','Positive'))
    plt.title('Attitutes: absolute values')
    plt.show()
    # in percentages
    pert = groupfre(alsent_cnt)
    pert_cum = pert.cumsum(axis=1)
    p4 = plt.bar(ind,pert_cum[:,2],color='r')
    p5 = plt.bar(ind,pert_cum[:,1],color='k')
    p6 = plt.bar(ind,pert_cum[:,0],color='g')
    plt.xticks(ind,('American','Delta','Southwest','US', 'United', 'Virgin'))
    plt.legend((p1[0], p2[0], p3[0]),('Negative','Neutral','Positive'))
    plt.title('Attitutes: percentages')
    plt.show()
      
def modeldata()
    pass
    

if __name__='__name__':
    tidydata()