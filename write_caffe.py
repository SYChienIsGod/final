# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:34:04 2015

@author: jan
"""

import paths
import numpy as np

import phonemes
import speakersent
import caffe_pb2

#%% Validation & Training data

# Select which groups should be in validation
fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))
rng = np.random.RandomState(0)
sentenceGroupId,sentenceGroupInst=speakersent.getSpeakerSentenceGroups(fbank_train_ids)
randomOrderSentIds = rng.permutation(list(set(sentenceGroupId)))

validationSetSize = 0
validationSet = list()
i = 0
while validationSetSize/float(len(sentenceGroupId)) < 0.2:
    validationSet.append(randomOrderSentIds[i])
    validationSetSize += sentenceGroupInst[randomOrderSentIds[i]]
    i+=1

validationSelection = np.in1d(sentenceGroupId,(validationSet))
trainingSelection = validationSelection==0

print 'Selected validation set: {}/{}'.format(validationSetSize,len(sentenceGroupId))

#%% Read FBANK Training Data

fbank_train = np.loadtxt(paths.pathToFBANKTrain,dtype='float32',delimiter=' ',usecols=range(1,70))
fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))
fbank_train_lab = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(1,))
fbank_train_lab_ids = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(0,))
fbank_train_labels = phonemes.trans_ph48(fbank_train_lab)

fbank_train_labels_sorted = np.zeros_like(fbank_train_labels)

oldLabels = dict(zip(fbank_train_lab_ids,fbank_train_labels))
for i in range(fbank_train_ids.shape[0]):
    fbank_train_labels_sorted[i]=oldLabels[fbank_train_ids[i]]

train_data = fbank_train[trainingSelection,:]
train_ids = fbank_train_ids[trainingSelection]
train_labels = fbank_train_labels_sorted[trainingSelection]

val_data = fbank_train[validationSelection,:]
val_ids = fbank_train_ids[validationSelection]
val_labels = fbank_train_labels_sorted[validationSelection]

test_data = np.loadtxt(paths.pathToFBANKTest,dtype='float32',delimiter=' ',usecols=range(1,70))
test_ids = np.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))
test_labels = np.zeros(shape=(test_data.shape[0],),dtype='int_')

print 'Loaded training, validation and test data.'

#%%

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(train_data)

train_data_std = scaler.transform(train_data)
val_data_std = scaler.transform(val_data)
test_data_std = scaler.transform(test_data)

print 'Scaled data.'

#%%

# Attention: Sentence Segments start with 1
def arrangeData(data,ids,before,after):
    arranged = np.zeros(shape=(data.shape[0],(before+after+1)*data.shape[1]),dtype='float32')
    sswi = speakersent.getSpeakerSentenceWordId(ids)
    w = data.shape[1]
    for i in range(data.shape[0]):
        sp,se,wo=speakersent.SpSeWo(ids[i])
        for k in range(-before,after+1):
            wo_=wo+k
            if wo_ < 1:
                wo_ = 1
            elif wo_ > len(sswi[sp][se]):
                wo_ = len(sswi[sp][se])
            arranged[i,(k+before)*w:(k+1+before)*w] = data[sswi[sp][se][wo_],:]
    return arranged
    
#%%

phonemes2Id = speakersent.getPhonemeIds(train_labels)

# Attention: Sentence Segments start with 1
def arrangeDataPhRep(data,ids,before,after,phonIds,labels):
    arranged = np.zeros(shape=(data.shape[0],(before+after+1)*data.shape[1]),dtype='float32')
    sswi = speakersent.getSpeakerSentenceWordId(ids)
    w = data.shape[1]
    for i in range(data.shape[0]):
        sp,se,wo=speakersent.SpSeWo(ids[i])
        for k in range(-before,after+1):
            wo_=wo+k
            if wo_ < 1:
                wo_ = 1
            elif wo_ > len(sswi[sp][se]):
                wo_ = len(sswi[sp][se])
            if np.random.rand(1)[0] < 0.4 and k != 0:
                lab = labels[sswi[sp][se][wo_]]
                tmpIdx=np.random.randint(0,len(phonIds[lab]))
                replaceIdx =phonIds[lab][tmpIdx]
                arranged[i,(k+before)*w:(k+1+before)*w] = data[replaceIdx,:]
            else:
                arranged[i,(k+before)*w:(k+1+before)*w] = data[sswi[sp][se][wo_],:]
    return arranged
                    

#%%

import leveldb

def writeData(data,labels,path,order=None):
    db = leveldb.LevelDB(path)
    batch = leveldb.WriteBatch()
    if type(order)==type(None):
        perm=np.arange(labels.size)
    else:
        perm=order
    for i in range(data.shape[0]):
        datum = caffe_pb2.Datum()
        datum.channels=1
        datum.height=1
        datum.width=data.shape[1]
        datum.float_data.extend(list(data[perm[i]].astype(float)))
        datum.label=labels[perm[i]]
        result = datum.SerializeToString()
        key = i
        batch.Put(str(key),result)
    db.Write(batch,sync=True)
    
#%% Clean the databases
    
trainDB = './caffedata/train.lvl'
valDB   = './caffedata/val.lvl'
testDB  = './caffedata/test.lvl'

import shutil

shutil.rmtree(trainDB)
shutil.rmtree(valDB)
shutil.rmtree(testDB)

#%% Write Data

randomTrainingOrder = rng.permutation(np.arange(train_data_std.shape[0]))
writeData(arrangeData(train_data_std,train_ids,4,1),train_labels,'./caffedata/train.lvl',randomTrainingOrder)
writeData(arrangeData(val_data_std,val_ids,4,1),val_labels,'./caffedata/val.lvl')
writeData(arrangeData(test_data_std,test_ids,4,1),test_labels,'./caffedata/test.lvl')
        
print 'Wrote data to databases.'
