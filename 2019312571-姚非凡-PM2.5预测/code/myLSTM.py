# 代码参考：https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# load train dataset
dataset = read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2015/All.csv', header=0, index_col=0)
dataset = dataset.drop(columns=["hour","AQI","PM10",'grade'])
order = ['PM2.5','CO','NO2','O3','SO2']
dataset = dataset[order]
values = dataset.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[6, 7, 8, 9]], axis=1, inplace=True)
thetrain = reframed
print(thetrain.head())

# load test dataset
dataset2 = read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2014/All.csv', header=0, index_col=0)
dataset2 = dataset2.drop(columns=["hour","AQI","PM10",'grade'])
order = ['PM2.5','CO','NO2','O3','SO2']
dataset2 = dataset2[order]
values2 = dataset2.values
values2 = values2.astype('float32')
# normalize features
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaled2= scaler2.fit_transform(values2)
reframed2 = series_to_supervised(scaled2, 1, 1)
reframed2.drop(reframed2.columns[[6, 7, 8, 9]], axis=1, inplace=True)
thetest = reframed2
print(thetest.head())

# split into train and test sets
traindata = thetrain.values
testdata = thetest.values
train = traindata
test = testdata
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.summary()
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_split=0.2, verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend()
pyplot.savefig('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/bare_jrnl_transmag/Loss.pdf')
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
# print(yhat)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
print(inv_yhat)

testgrade = []
number = inv_yhat
for i in range(len(number)):
    if number[i]>=0 and number[i] < 35:
        testgrade.append(1)
    elif number[i]>=35 and number[i] < 75:
        testgrade.append(2)
    elif number[i]>=75 and number[i] < 115:
        testgrade.append(3)
    elif number[i]>=115 and number[i] < 150:
        testgrade.append(4)
    elif number[i]>=150 and number[i] < 250:
        testgrade.append(5)
    elif number[i]>=250:
        testgrade.append(6)
testgrade = np.array(testgrade)

dataset3 = read_csv('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2014/All.csv', header=0, index_col=0)
trueGrade = dataset3['grade']
trueGrade = trueGrade.values[0:-1]


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#labels表示你不同类别的代号
labels = ['1','2','3','4','5','6']
y_true = trueGrade
y_pred = testgrade
cm = confusion_matrix(y_true, y_pred)

import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm, classes=labels, normalize=True, title='Normalized confusion matrix')
plt.savefig('/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/bare_jrnl_transmag/cm.pdf',bbox_inches = 'tight')
plt.show()