import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import tensorflow as tf
from sklearn.svm import SVC

# loading data
train = pd.read_csv('data/bikeshareTraining.csv', sep=',', index_col=['trip_id'])
temp = pd.read_csv('data/temperature1.csv', sep=',', index_col=['Date'], parse_dates=True)
train = train.head(80000)

# preprocess data

# check weekday
def isWeekday(row):
    weekno = row['start_time'].weekday()
    if weekno < 5:
        return 1
    else:
        return 0


# return the start hour for each record
def hour(row):
    weekhour = row['start_time'].hour
    return weekhour


# change to numeric type for training
train['tripduration'] = train['tripduration'].str.replace(',', '')
train[["bikeid", "tripduration", "from_station_id", "to_station_id"]] = train[
    ["bikeid", "tripduration", "from_station_id", "to_station_id"]].apply(pd.to_numeric)

# drop the rows which duration larger than 10 hours
train = train.drop(train[train.tripduration > 36000].index)

# use dummy type to deal with usertype
usertype_dummies = pd.get_dummies(train['usertype'])
train = train.join(usertype_dummies)
train.drop(['usertype'], axis=1, inplace=True)

# change to datetime and perform feature adding
train[["start_time", "end_time"]] = train[["start_time", "end_time"]].apply(pd.to_datetime)
train['isWeekday'] = train.apply(isWeekday, axis=1)
train['start_hour'] = train.apply(hour, axis=1)


# add teperature
def addTemp(row):
    dateTemp = row['start_time'].date()
    aveTemp = temp.loc[dateTemp]['Average']
    # rain = temp.loc[dateTemp]['Precipitation']
    return aveTemp


# add precipitation
def addRain(row):
    dateTemp = row['start_time'].date()
    # aveTemp = temp.loc[dateTemp]['Average']
    rain = temp.loc[dateTemp]['Precipitation']
    return rain


train['temperature'] = train.apply(addTemp, axis=1)
train['precipitation'] = train.apply(addRain, axis=1)


# assume the speed of riding bikes is 10 km/h
# the max distance for each row is speed * duration_time
def getDistance(row):
    return row['tripduration'] * 10 / 3600 * 1.0


train['max_dist'] = train.apply(getDistance, axis=1)

# use dummy type to deal with gender
gender_dummies = pd.get_dummies(train['gender'])
train = train.join(gender_dummies)
# train.drop(['gender'], axis=1,inplace=True)
# train.info()

# decision tree model

X = train[[ "bikeid", "start_hour", "isWeekday", "tripduration", "from_station_id", "Customer", "Subscriber", \
           "max_dist", "temperature", "precipitation", "Female", "Male"]]
Y = train["to_station_id"]
names = ["bikeid", "start_hour", "isWeekday", "tripduration", "from_station_id", "Customer", "Subscriber", \
         "max_dist", "temperature", "precipitation", "Female", "Male"]

# 70% as trainning, 30% as test validation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

def dectree():
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    #acc = r2_score(y_test, rf.predict(X_test))
    acc = precision_score(y_test, clf.predict(X_test), average='micro')
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True))
    print(acc)


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

# nntrain(X_train, y_train,X_test, y_test)

def svmmethod():
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    SVMprecision=clf.score(X_test, y_test)
    print(SVMprecision)

def redomforestmethod():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=18, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=None,
                oob_score=False, random_state=0, verbose=0, warm_start=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    # X_train1, X_test1, y_train1, y_test1 = train_test_split(X_test, y_test, test_size=0.1, random_state=42)
    rf.fit(X_train, y_train)
    acc = precision_score(y_test, rf.predict(X_test), average='micro')
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))
    print(acc)


# redomforestmethod()
dectree()
from sklearn.ensemble import AdaBoostClassifier
bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME", n_estimators=200, learning_rate=0.8)
bdt.fit(X_train, y_train)
#X_train1, X_test1, y_train1, y_test1 = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
acc = precision_score(y_test, bdt.predict(X_test), average='micro')
print(acc)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), bdt.feature_importances_), names), reverse=True))