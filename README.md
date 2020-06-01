# PM2.5-homework

## 文件夹codes
全部代码
## 文件夹data
用来测试的2014年测试数据
## 软件环境与软件版本
macOS(系统不同相对路径格式需要更改) 
Python 3.6 
## Python 库
math  
numpy  
pandas  
sklearn  
os  
keras  
matplotlib  
datetime  
csv
tensorflow
## 整理数据的命令
将数据放入beijing_20140101-20141231文件夹后，需先运行打开终端输入：  
cd 拖入下载的文件夹（2019312571-姚非凡-PM2.5预测）  
然后运行：  
python ./code/theNanValue.py  
再运行：  
python ./code/dataClean.py  
从而对原始数据进行数据清洗和特征提取。
## 模型
模型保存为在code中的model.h5
## 训练模型
python ./code/ModeltoTest.py  
将会打印结果。
