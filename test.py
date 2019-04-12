import pandas as pd
import numpy as np
import datetime
import time

def string2timestamp(strValue):
    try:
        d = datetime.datetime.strptime(strValue, "%Y-%m-%d %H:%M:%S")
        t = d.timetuple()
        timeStamp = int(time.mktime(t))
        timeStamp = float(str(timeStamp) + str("%06d" % d.microsecond)) / 1000000
        return timeStamp
    except ValueError as e:
        print(e)
        return 0

def string2hour(strValue):
    try:
        d = datetime.datetime.strptime(strValue, "%Y-%m-%d %H:%M:%S")
        return d.hour
    except ValueError as e:
        print(e)
        return 0

def string2min(strValue):
    try:
        d = datetime.datetime.strptime(strValue, "%Y-%m-%d %H:%M:%S")
        return d.minute
    except ValueError as e:
        print(e)
        return 0


def deletewronttime():
    trainData["start_time_stamp"] = trainData["start_time"].apply(lambda x: string2timestamp(x))
    trainData["end_time_stamp"] = trainData["end_time"].apply(lambda x: string2timestamp(x))
    trainData["durationcheck"] = trainData["end_time_stamp"] - trainData["start_time_stamp"]
    trainData["durationdiff"] = trainData["durationcheck"] - trainData["tripduration"]
    print(trainData[(trainData[u'durationdiff'] != 0.0)])

trainData=pd.read_csv('data/bikeshareTraining.csv')
#
pd.options.display.max_columns = 20
print(string2timestamp(trainData.head(5)["start_time"][0]))
trainData["tripduration"] = trainData["tripduration"].apply(lambda x: x.replace(',', ''))
trainData["tripduration"] = trainData["tripduration"].apply(lambda x: float(x))
trainData["start_time_hour"] = trainData["start_time"].apply(lambda x: string2hour(x))
# trainData["end_time_hour"] = trainData["end_time"].apply(lambda x: string2hour(x))
# trainData["start_time_minute"] = trainData["start_time"].apply(lambda x: string2min(x))

trainData["gender"].fillna('None', inplace=True)
trainData["birthyear"].fillna(0, inplace=True)
trainData = trainData.replace({'gender': {"Male" : 1, "Female" : 2, "None" : 3}})
trainData = trainData.replace({'usertype': {"Customer" : 1, "Subscriber" : 2}})

label=trainData['to_station_id']
trainData = trainData.drop(columns=['trip_id','from_station_name','to_station_name', 'start_time', 'end_time', 'to_station_id'])