import pandas as pd
import numpy as np
import random
import datetime
from haversine import haversine, Unit
import argparse
import pickle

#Read the Pollution Data and return the dataframe
def readPollutionData(loc = '20_10_all2.csv', day = -1, nr = 0.2):
    #consider a day's worth of data
    df = pd.read_csv(loc)
    #type casting
    mfactor = 1/nr
    df.pm1_0 = df.pm1_0.astype(float)
    df.pm2_5 = df.pm2_5.astype(float)
    df.pm10 = df.pm10.astype(float)
    df.lat = round(round(mfactor*df.lat.astype(float),2)/mfactor,3)
    df.long= round(round(mfactor*df.long.astype(float),2)/mfactor,3)
    df.pressure = df.pressure.astype(float)
    df.temperature = df.temperature.astype(float)
    df.humidity = df.humidity.astype(float)
    df.dateTime = pd.to_datetime(df.dateTime)
    if(day!=-1):
        df = df[df.dateTime.dt.day == day]
    #print("length of Dataset:",len(df))
    return df 
    #Ensuring Delhi region and removing outliers from data
    # df = df[(df.lat.astype(int) == 28) &(df.long.astype(int) == 77)]
    # df = df[(df.pm1_0<=1500) & (df.pm2_5<=1500) & (df.pm10<=1500) & (df.pm1_0>=20) & (df.pm2_5>=30) & (df.pm10>=30)]
    # df = df[(df.humidity<=60)&(df.humidity>=7)]

#DEBUGGING -- only to check the number of data points in each hour.
def hourlyDistribution(df, noPrint = False):
    num = [ len(df[(df.dateTime.dt.hour == i)]) for i in range(24) ]
    if(not noPrint):
        print(num)
    return num

# roundOff dataPoints to 15min and remove the clutter from dataframe, keeping only dateTime, lat, long, pollutant.
def roundOff(df, roundTime = '15min', pollutant = ['pm10']):
    dfHour = df[['dateTime','lat','long']+pollutant]
    #--------------------------------Rounding @15min----------------------------------
    dfHour.dateTime = dfHour.dateTime.dt.round('15min')
    #---------------------------------Only PM10---------------------------------------
    meaned = dfHour.groupby(['dateTime','lat','long']).mean().reset_index()
    meaned[pollutant] = round(meaned[pollutant],2)
    return meaned

# dataframe -> numpy, also checks for time interval
def timeShredding(df,startMin = 300, endMin = 1350):
    #Converting Time to minutes
    meaned = df.copy(deep = True)
    meaned.dateTime = meaned.dateTime.dt.hour*60+meaned.dateTime.dt.minute
    meaned.dateTime %= 1440
    meaned = meaned[(meaned.dateTime>=startMin) & (meaned.dateTime<=endMin)]
    meaned = meaned.sort_values(by = ['dateTime','lat','long'])
    data = meaned.to_numpy()
    return data

#get the testData from the train numpy data.
def testSampling(data, percent = 0.05, makeACopy = False, copyName = "sampleTestIndices.txt", takeFromCopy=False):
    dataTemp = data*1.0
    testSet = []
    if(takeFromCopy):
        testSet = list(np.loadtxt(copyName, delimiter=',', dtype=int))
    else: 
        testSet = random.sample(range(dataTemp.shape[0]),int(dataTemp.shape[0]*percent))
        if(makeACopy):
            np.savetxt(copyName, testSet, delimiter=',')
    testData = dataTemp[testSet]
    dataTemp = np.delete(dataTemp,testSet,0)
    return testData, dataTemp