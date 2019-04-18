import pandas as pd
import numpy as np
import datetime
import time
import tensorflow as tf


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


def deletewronttime(trainData):
    trainData["start_time_stamp"] = trainData["start_time"].apply(lambda x: string2timestamp(x))
    trainData["end_time_stamp"] = trainData["end_time"].apply(lambda x: string2timestamp(x))
    trainData["durationcheck"] = trainData["end_time_stamp"] - trainData["start_time_stamp"]
    trainData["durationdiff"] = trainData["durationcheck"] - trainData["tripduration"]
    print(trainData[(trainData[u'durationdiff'] != 0.0)])


def mainp(total_size,train_rate):
    train_size=int(total_size*train_rate)
    trainData = pd.read_csv('data/bikeshareTraining.csv')
    #
    trainData=trainData.head(total_size)

    pd.options.display.max_columns = 20
    print(string2timestamp(trainData.head(5)["start_time"][0]))
    trainData["tripduration"] = trainData["tripduration"].apply(lambda x: x.replace(',', ''))
    trainData["tripduration"] = trainData["tripduration"].apply(lambda x: float(x))
    trainData["start_time_hour"] = trainData["start_time"].apply(lambda x: string2hour(x))
    # trainData["end_time_hour"] = trainData["end_time"].apply(lambda x: string2hour(x))
    # trainData["start_time_minute"] = trainData["start_time"].apply(lambda x: string2min(x))

    trainData["gender"].fillna('None', inplace=True)
    trainData["birthyear"].fillna(0, inplace=True)
    trainData = trainData.replace({'gender': {"Male": 1, "Female": 2, "None": 3}})
    trainData = trainData.replace({'usertype': {"Customer": 1, "Subscriber": 2}})

    # print(trainData.loc[trainData['tripduration'].isnull()])

    trainData = trainData.drop(
        columns=['trip_id','bikeid', 'from_station_name', 'to_station_name', 'start_time', 'end_time'])

    # trainData=trainData.apply(lambda x: float(x))

    label = trainData['to_station_id']
    trainData = trainData.drop(columns='to_station_id')

    trainData=trainData.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    x_train = trainData.head(train_size)
    y_train = label.head(train_size)
    x_test = trainData.tail(total_size-train_size)
    y_test = label.tail(total_size-train_size)
    return x_train, y_train,x_test, y_test

x_train, y_train,x_test, y_test=mainp(20000,0.7)
print(x_train.head(5))
# print(y_train.head(5))

X_train=np.array(x_train,dtype=np.float)
Y_train=np.array(y_train,dtype=np.int)
X_test=np.array(x_test,dtype=np.float)
Y_test=np.array(y_test,dtype=np.int)


model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(650, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=8)
model.evaluate(X_test, Y_test)