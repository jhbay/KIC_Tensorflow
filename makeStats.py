
import pandas as pd
from statistics import *

def getStats(dataList, modelName, listSize):
    minList = []
    maxList = []
    meanList = []
    idxList = []
    for i in range(0, len(dataList), listSize): # start, stop, step
        partList = dataList[i:i + listSize]
        # print('index::{} ~ {} // mean: {}'.format(i, i + listSize, partMean ) )
        minList.append(min(partList))
        maxList.append(max(partList))
        meanList.append(mean(partList))
        idxList.append(i + listSize)
    statResult = pd.DataFrame({modelName+'_min':minList, modelName+'_max':maxList, modelName+'_mean':meanList }, index = idxList)
    # print('statResult::', statResult)
    return statResult

def getStatSet(dataDict, fileName, listSize):
    statSetResult = pd.DataFrame()
    for modelName in dataDict.keys():
        # print('dataDict[modelName]:{}, modelName:{}, listSize:{}'.format(len(dataDict[modelName]), modelName, listSize))
        statSetResult = pd.concat([statSetResult, getStats(dataDict[modelName], modelName, listSize)], axis=1)

    print('statSetResult::', statSetResult)
    statSetResult.to_csv(fileName + "_Result.csv", mode='w')

def getSampleList():
    sampleStr = "0.1, 0.13, 0.13, 0.23999999, 0.19, 0.34999999, 0.47999999, 0.38, 0.43000001, 0.41999999, 0.47999999, 0.47, 0.57999998, 0.68000001, 0.57999998, 0.72000003, 0.69, 0.64999998, 0.69999999, 0.72000003, 0.69999999, 0.66000003, 0.80000001, 0.76999998, 0.77999997, 0.81999999, 0.79000002, 0.80000001, 0.85000002, 0.82999998, 0.73000002, 0.83999997, 0.82999998, 0.75999999, 0.86000001, 0.85000002, 0.91000003, 0.86000001, 0.88999999, 0.91000003, 0.88999999, 0.88999999, 0.91000003, 0.89999998, 0.94, 0.83999997, 0.81, 0.91000003, 0.86000001, 0.83999997, 0.94, 0.89999998, 0.91000003, 0.87, 0.94999999, 0.89999998, 0.94, 0.85000002, 0.87, 0.91000003, 0.91000003, 0.94, 0.92000002, 0.91000003, 0.91000003, 0.89999998, 0.95999998, 0.85000002, 0.89999998, 0.92000002, 0.94999999, 0.87, 0.86000001, 0.94, 0.88999999, 0.92000002, 0.93000001, 0.89999998, 0.91000003, 0.94999999, 0.92000002, 0.94999999, 0.88999999, 0.92000002, 0.89999998, 0.94, 0.94999999, 0.93000001, 0.94, 0.91000003, 0.97000003, 0.94999999, 0.95999998, 0.91000003, 0.94999999, 0.89999998, 0.94999999, 0.97000003, 0.95999998, 0.92000002, 0.97000003"
    sampleList = [float(x.strip()) for x in sampleStr.split(',')]
    # print('len(sampleList)::',len(sampleList))
    return sampleList

# getStats(getSampleList(), '../python_DL/cnn_logs/nowTime', 25)
dataDict = {'train':getSampleList(),'test':getSampleList()}
getStatSet(dataDict, "../python_DL/cnn_logs/time", 25)






