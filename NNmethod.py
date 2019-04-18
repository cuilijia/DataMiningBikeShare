import pandas as pd
import numpy as np
import datetime
import time
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

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

def string2day(strValue):
    try:
        d = datetime.datetime.strptime(strValue, "%Y-%m-%d %H:%M:%S")
        return d.day
    except ValueError as e:
        print(e)
        return 0

def getweeken(strValue):
    day=string2day(strValue)
    if int(day) in [1,7,8,14,15]:
        return 1
    else:
        return 0


def deletewronttime(trainData):
    trainData["start_time_stamp"] = trainData["start_time"].apply(lambda x: string2timestamp(x))
    trainData["end_time_stamp"] = trainData["end_time"].apply(lambda x: string2timestamp(x))

    trainData["durationcheck"] = trainData["end_time_stamp"] - trainData["start_time_stamp"]
    trainData["durationdiff"] = trainData["durationcheck"] - trainData["tripduration"]
    # print(trainData[(trainData[u'durationdiff'] != 0.0)]['durationdiff'])
    trainData = trainData[(trainData[u'durationdiff'] == 0.0)]
    trainData = trainData[(trainData['tripduration'] <20000)]
    # print(trainData[(trainData[u'durationdiff'] != 0.0)]['durationdiff'])
    trainData = trainData.drop(columns=['durationcheck', 'durationdiff', 'start_time_stamp', 'end_time_stamp'])
    return trainData


def mainp(total_size, train_rate):
    train_size = int(total_size * train_rate)
    trainData = pd.read_csv('data/bikeshareTraining.csv')
    #
    trainData = trainData.head(total_size)

    pd.options.display.max_columns = 20
    print(string2timestamp(trainData.head(5)["start_time"][0]))

    trainData["gender"].fillna('None', inplace=True)
    trainData["birthyear"].fillna(2018.0, inplace=True)

    trainData["age"] = 2018.0-trainData["birthyear"]
    trainData["tripduration"] = trainData["tripduration"].apply(lambda x: x.replace(',', ''))
    trainData["tripduration"] = trainData["tripduration"].apply(lambda x: float(x))
    trainData["start_time_hour"] = trainData["start_time"].apply(lambda x: string2hour(x))
    trainData["end_time_hour"] = trainData["end_time"].apply(lambda x: string2hour(x))
    trainData["start_time_minute"] = trainData["start_time"].apply(lambda x: string2min(x))
    trainData["end_time_minute"] = trainData["start_time"].apply(lambda x: string2min(x))
    # trainData["isweeken"] = trainData["start_time"].apply(lambda x: getweeken(x))


    trainData = trainData.replace({'gender': {"Male": 1, "Female": 2, "None": 3}})
    trainData = trainData.replace({'usertype': {"Customer": 1, "Subscriber": 2}})

    trainData=deletewronttime(trainData)
    # print(trainData.loc[trainData['tripduration'].isnull()])

    # # coralation!!
    # trainData.corr()
    # ax = sns.heatmap(trainData.corr(), linewidths=.5)
    # plt.show()

    trainData = trainData.drop(
        columns=['trip_id', 'from_station_name', 'to_station_name', 'start_time', 'end_time', 'end_time'])
    # trainData = trainData.drop(
    #     columns=['trip_id', 'from_station_name', 'to_station_name', 'start_time', 'end_time', 'bikeid', 'usertype',
    #              'gender', 'birthyear', 'from_station_id', 'start_time_hour', 'end_time_hour'])

    # trainData=trainData.apply(lambda x: float(x))

    label = trainData['to_station_id']
    trainData = trainData.drop(columns='to_station_id')

    x_train = trainData.head(train_size)
    y_train = label.head(train_size)
    x_test = trainData.tail(total_size - train_size)
    y_test = label.tail(total_size - train_size)

    x_train = x_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    x_test = x_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    return x_train, y_train, x_test, y_test


def nntrain(x_train, y_train, x_test, y_test):
    X_train = np.array(x_train, dtype=np.float)
    Y_train = np.array(y_train, dtype=np.int)
    X_test = np.array(x_test, dtype=np.float)
    Y_test = np.array(y_test, dtype=np.int)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30, activation=tf.nn.relu),
        tf.keras.layers.Dense(30, activation=tf.nn.relu),
        tf.keras.layers.Dense(30, activation=tf.nn.relu),
        # tf.keras.layers.Dense(30, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(700, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10)
    model.evaluate(X_test, Y_test)


x_train, y_train, x_test, y_test = mainp(10000, 0.7)
print(x_train.head(5))
nntrain(x_train, y_train,x_test, y_test)