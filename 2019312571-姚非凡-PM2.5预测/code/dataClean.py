import pandas as pd
import csv
import os

def mainF():
    path = os.path.abspath('..')
    filename1 = pd.read_csv(path+'/data/beijing_20140101-20141231/F1.csv',header=None, names = ['date','hour','pollution','value'])
    # filename2 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2015/F2.csv',header=None, names = ['date','hour','pollution','value'])
    # filename3 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2016/F3.csv',header=None, names = ['date','hour','pollution','value'])
    # filename4 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2016/F4.csv',header=None, names = ['date','hour','pollution','value'])
    # filename5 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2017/F5.csv',header=None, names = ['date','hour','pollution','value'])

    frames = [filename1]
    result = pd.concat(frames)
    final = result.pivot_table(columns="pollution", values="value", aggfunc="first",index=("date", 'hour')).reset_index()
    print(final)
    final.to_csv(path+'/data/beijing_20140101-20141231/Feature.csv', index=False, header=True)

def mainL():
    path = os.path.abspath('..')
    filename1 = pd.read_csv(path+'/data/beijing_20140101-20141231/L1.csv',header=None, names=['date', 'hour', 'air', 'value'])
    # filename2 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2015/L2.csv',header=None, names=['date', 'hour', 'air', 'value'])
    # filename3 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2016/L3.csv',header=None, names=['date', 'hour', 'air', 'value'])
    # filename4 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2016/L4.csv',header=None, names=['date', 'hour', 'air', 'value'])
    # filename5 = pd.read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2017/L5.csv',header=None, names=['date', 'hour', 'air', 'value'])

    frames = [filename1]
    result = pd.concat(frames)
    final = result.pivot_table(columns="air", values="value", aggfunc="first",index=("date", 'hour')).reset_index()
    print(final)
    final.to_csv(path+'/data/beijing_20140101-20141231/label.csv', index=False, header=True)

def mainFL():
    path = os.path.abspath('..')
    filename1 = pd.read_csv(path+'/data/beijing_20140101-20141231/Feature.csv')
    filename2 = pd.read_csv(path+'/data/beijing_20140101-20141231/label.csv')
    filename = pd.concat([filename1, filename2], axis=1)  # concat无how
    F = filename.drop_duplicates().T.drop_duplicates().T
    F = F.dropna(axis=0)
    F.to_csv(path+'/data/beijing_20140101-20141231/temp.csv', index=False,
             header=True)
    All = pd.read_csv(path+'/data/beijing_20140101-20141231/temp.csv')
    grade = []
    number = All['PM2.5']
    for i in range(len(number)):
        if number[i]>=0 and number[i] < 35:
            grade.append(1)
        elif number[i]>=35 and number[i] < 75:
            grade.append(2)
        elif number[i]>=75 and number[i] < 115:
            grade.append(3)
        elif number[i]>=115 and number[i] < 150:
            grade.append(4)
        elif number[i]>=150 and number[i] < 250:
            grade.append(5)
        elif number[i]>=250:
            grade.append(6)
    g = pd.DataFrame(grade, columns=['grade'])
    FF = pd.concat([All, g], axis=1)
    FF.to_csv(path+'/data/beijing_20140101-20141231/All.csv', index=False,
             header=True)

if __name__ == '__main__':
    mainF()
    mainL()
    mainFL()
