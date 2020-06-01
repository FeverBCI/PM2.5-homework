# -*- coding: UTF-8 -*-
import pandas as pd
import datetime
import csv
import os

def writer_data_extra(date, hour, type, mean):
    path = os.path.abspath('..')
    csvfile = open(path+'/data/beijing_20140101-20141231/F1.csv', 'a', newline='')
    writer = csv.writer(csvfile)
    info = [date, hour, type, mean]
    writer.writerow(info)
    csvfile.close()

def writer_data_all(date, hour, type, mean):
    path = os.path.abspath('..')
    csvfile = open(path+'/data/beijing_20140101-20141231/L1.csv', 'a', newline='')
    writer = csv.writer(csvfile)
    info = [date, hour, type, mean]
    writer.writerow(info)
    csvfile.close()


def run_extra():
    path = os.path.abspath('..')
    begin = datetime.date(2014, 4, 29)
    end = datetime.date(2014, 12, 30)
    d = begin
    delta = datetime.timedelta(days=1)
    q = 0
    while d <= end:
        num = d.strftime('%m%d')
        filename = pd.read_csv(path+'/data/beijing_20140101-20141231/beijing_extra_2014' + num + '.csv')
        four=[0,2,4,5]
        # for j in range(0, 8, 2):
        # for j in range(0,4):
        nff = filename[0:len(filename):2]
        for j in range(0, int(len(filename)/2)):
            nf = nff[j:j+1]
            # nf = filename[four[j]::7]
            # 奇数的语句print x[::2]
            # 偶数的语句print x[1::2]
            for i in nf.columns[3:]:
                a = nf[str(i)].median()
                nf.fillna(a, inplace=True)
            date = list(set(nf['date']))[0]
            hour = list(set(nf['hour']))[0]
            type = list(set(nf['type']))[0]
            sum = 0
            for i in nf.columns[3:]:
                b = nf[str(i)].mean()
                sum += b
            mean = round(sum / len(nf.columns[3:]), 1)
            # print('date：{} type:{} val:{}'.format(date, type, mean))
            writer_data_extra(date, hour, type, mean)
            q += 1
            if q % 200 == 0:
                print("正在转录...")
                print(d)
        d += delta

    print("**********转录完毕**************")

def run_all():
    path = os.path.abspath('..')
    begin = datetime.date(2014, 4, 29)
    end = datetime.date(2014, 12, 30)
    d = begin
    delta = datetime.timedelta(days=1)
    q = 0
    while d <= end:
        num = d.strftime('%m%d')
        filename = pd.read_csv(path+'/data/beijing_20140101-20141231/beijing_all_2014' + num + '.csv')
        filename = filename[~filename['type'].isin(['PM2.5_24h'])]
        filename = filename[~filename['type'].isin(['PM10_24h'])]
        for j in range(0, len(filename)):
        # for j in range(0,4):
            nf = filename[j:j+1]
            # nf = filename[four[j]::7]
            # 奇数的语句print x[::2]
            # 偶数的语句print x[1::2]
            for i in nf.columns[3:]:
                a = nf[str(i)].median()
                nf.fillna(a, inplace=True)
            date = list(set(nf['date']))[0]
            hour = list(set(nf['hour']))[0]
            type = list(set(nf['type']))[0]
            sum = 0
            for i in nf.columns[3:]:
                b = nf[str(i)].mean()
                sum += b
            mean = round(sum / len(nf.columns[3:]), 1)
            # print('date：{} type:{} val:{}'.format(date, type, mean))
            writer_data_all(date, hour, type, mean)
            q += 1
            if q % 200 == 0:
                print("正在转录...")
                print(d)
        d += delta

    print("**********转录完毕**************")

if __name__ == '__main__':
    run_extra()
    run_all()
    os.system('say "your program has finished"')