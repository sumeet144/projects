import numpy as np
import csv
import pandas as pd
from pandas import *
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def cleandata():
    train = pd.read_csv('./train.csv')
    feature = pd.read_csv('./features.csv')
    test = pd.read_csv('./test.csv')


    feature = del_markdown(feature)

    feature = del_unemployment(feature)
    #feature = feature

    train = del_train_markdown(train)

    return (train,test,feature)

def del_markdown(feature):
    '''
    a = notnull(feature.MarkDown1)
    b = notnull(feature.MarkDown2)
    c = notnull(feature.MarkDown3)
    d = notnull(feature.MarkDown4)
    e = notnull(feature.MarkDown5)
    train = feature[a|b|c|d|e]
    '''
    feature = feature[feature.Date >= '2011-11-04']
    return feature

def del_unemployment(feature):
    feature = feature[['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','IsHoliday']]
    return feature

def del_train_markdown(train):
    train = train[train.Date >= '2011-11-11']
    return train

def combi_train_feature(train,test,feature,markdown):
    train = np.array(train)
    test = np.array(test)
    feature = np.array(feature)
    train_x,train_y,test_x,dates=[],[],[],[]
    j=0
    for i in range(len(train)):
        train_x.append([])
        store,dept,date,sales,isholiday = train[i]

        f = find_from_feature(store,date,feature,markdown)
        train_y.append(sales)
        train_x[j] =list(f)

        temp = date.split('-')
        y,m,d =int(temp[0]),int(temp[1]),int(temp[2])
        ymd = datetime.date(y,m,d)
        week = datetime.timedelta(days=7)
        preweek = ymd-week
        preweek = str(preweek)
        nextweek = ymd+week
        nextweek = str(nextweek)
        preweek = get_holiday_feature(preweek)
        thisweek = get_holiday_feature(date)
        nextweek = get_holiday_feature(nextweek)
        train_x[j] =train_x[j]+preweek+thisweek+nextweek
        j += 1
    j = 0
    for i in range(len(test)):
        test_x.append([])
        store,dept,date,isholiday = test[i]
        f = find_from_feature(store,date,feature,markdown)
        test_x[j] = list(f)

        temp = date.split('-')
        y,m,d = int(temp[0]),int(temp[1]),int(temp[2])
        ymd = datetime.date(y,m,d)
        week = datetime.timedelta(days=7)
        preweek = ymd-week
        preweek = str(preweek)
        nextweek = ymd+week
        nextweek = str(nextweek)
        preweek = get_holiday_feature(preweek)
        thisweek = get_holiday_feature(date)
        nextweek = get_holiday_feature(nextweek)
        test_x[j] =test_x[j]+ preweek+thisweek+nextweek
        dates.append(date)
        j += 1
    return (train_x,train_y,test_x,dates)

def find_from_feature(store,date,feature,markdown):
    for i in range(len(feature)):
        if feature[i][0] == store and feature[i][1] == date:
            for j in range(4,9):
                if isnull(feature[i][j]):
                    feature[i][j] = markdown[j-4]
            return feature[i][2:-1]

def linear_model(train_x,train_y,test_x):
    clf = LinearRegression()
    clf.fit(train_x,train_y)
    test_y = clf.predict(test_x)
    return test_y

def decision_tree(train_x, train_y, test_x):
    regr = DecisionTreeRegressor(max_depth = 10)
    train_ynew = [str(x) for x in train_y]
    regr.fit(train_x, train_ynew)
    test_y = regr.predict(test_x)
    return test_y

def knn_model(train_x,train_y,test_x,k):
    #clf = KNeighborsClassifier(n_neighbors=k,algorithm='kd_tree')
    test_x = np.array(test_x)
    clf = KNeighborsClassifier(n_neighbors=k,algorithm='auto')
    train_ynew = [str(x) for x in train_y]
    clf.fit(train_x,train_ynew)
    test_y = clf.predict(test_x)
    return test_y

def nan_rep(trains):
    md = []
    md.append(list(trains.MarkDown1))
    md.append(list(trains.MarkDown2))
    md.append(list(trains.MarkDown3))
    md.append(list(trains.MarkDown4))
    md.append(list(trains.MarkDown5))
    result = []
    for m in md:
        temp = np.array([i for i in m if notnull(i)])
        result.append(temp.mean())
    return result


def get_holiday_feature(date):
    super_bowl = ['2010-02-12','2011-02-11','2012-02-10','2013-02-08']
    labor = ['2010-09-10','2011-09-09','2012-09-07','2013-09-06']
    thx = ['2010-11-26','2011-11-25','2012-11-23','2013-11-29']
    chris = ['2010-12-31','2011-12-30','2012-12-28','2013-12-27']
    if date in super_bowl:
        return [0,0,0,1]
    elif date in labor:
        return [0,0,1,0]
    elif date in thx:
        return [0,1,0,0]
    elif date in chris:
        return [1,0,0,0]
    else:
        return [0,0,0,0]


def write(y,store,dept,dates):
    f = open('./result.csv','a')
    for i in range(len(y)):
        Id = str(store)+'_'+str(dept)+'_'+str(dates[i])
        sales = y[i]
        f.write('%s,%s\n'%(Id,sales))
    f.close()

if __name__=="__main__":
    f = open('./result.csv','wb')
    f.write('Id,Weekly_Sales\n')
    f.close()
    train,test,feature = cleandata()
    for i in range(1,46):
        traindata = train[train.Store == i]
        testdata = test[test.Store == i]
        featuredata = feature[feature.Store == i]
        depts = list(set(traindata.Dept.values))
        dept_test = list(set(testdata.Dept.values))
        for dept in dept_test:
            if dept not in depts:
                #print i,dept
                tests = testdata[testdata.Dept == dept]
                dates = list(tests.Date)
                y=[0 for j in range(len(tests))]
                write(y,i,dept,dates)

        for dept in depts:
            trains = traindata[traindata.Dept == dept]
            tests = testdata[testdata.Dept == dept]

            markdown = nan_rep(featuredata)
            #print i,dept
            train_x,train_y,test_x,dates = combi_train_feature(trains,tests,featuredata,markdown)
            #print len(train_x),len(test_x)
            k = 3
            #print len(train_x),len(test_x)
            if len(test_x) > 0:
                if len(train_x) <k:
                    #test_y = knn_model(train_x,train_y,test_x,len(train_x))
                    test_y = decision_tree(train_x, train_y, test_x)
                    #test_y = linear_model(train_x,train_y,test_x)
                    write(test_y,i,dept,dates)
                else:
                    #test_y = knn_model(train_x,train_y,test_x,k)
                    test_y = decision_tree(train_x, train_y, test_x)
                    #test_y = linear_model(train_x,train_y,test_x)
                    write(test_y,i,dept,dates)
