# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:34:04 2015

@author: jan
"""

import paths
import numpy as np

import phonemes
import speakersent
import write_lvldb

NBefore = 5
NAfter = 5

DBPrefix = 'fbankD2_7-3'

#%% Validation & Training data

# Select which groups should be in validation
fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))

speakerSets=speakersent.getSpeakerSets(fbank_train_ids)
sentenceGroupId,sentenceGroupInst=speakersent.getSpeakerSentenceGroups(fbank_train_ids,speakerSets)
#randomOrderSentIds = rng.permutation(list(set(sentenceGroupId)))

#validationSetSize = 0
#validationSet = list()
#i = 0
#while validationSetSize/float(len(sentenceGroupId)) < 0.2:
#    validationSet.append(randomOrderSentIds[i])
#    validationSetSize += sentenceGroupInst[randomOrderSentIds[i]]
#    i+=1

validationSet = speakersent.getValidationSet(fbank_train_ids,sentenceGroupId,sentenceGroupInst)

validationSelection = np.in1d(sentenceGroupId,(validationSet))
trainingSelection = validationSelection==0

print 'Selected validation set: {}/{}'.format(np.count_nonzero(validationSelection),len(sentenceGroupId))

#%% Read FBANK Training Data

fbank_train = np.loadtxt(paths.pathToFBANKTrain,dtype='float32',delimiter=' ',usecols=range(1,70))
fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))
train_lab = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(1,))
train_lab_ids = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(0,))
train_labels = phonemes.trans_ph48(train_lab)

train_labels_sorted = np.zeros_like(train_labels)

oldLabels = dict(zip(train_lab_ids,train_labels))
for i in range(fbank_train_ids.shape[0]):
    train_labels_sorted[i]=oldLabels[fbank_train_ids[i]]

train_data = fbank_train[trainingSelection,:]
train_ids = fbank_train_ids[trainingSelection]
train_labels = train_labels_sorted[trainingSelection]

val_data = fbank_train[validationSelection,:]
val_ids = fbank_train_ids[validationSelection]
val_labels = train_labels_sorted[validationSelection]

test_data = np.loadtxt(paths.pathToFBANKTest,dtype='float32',delimiter=' ',usecols=range(1,70))
test_ids = np.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))
test_labels = np.zeros(shape=(test_data.shape[0],),dtype='int_')

print 'FBANK: Loaded training, validation and test data.'

#%% Read FBANK Delta2 Training Data

#fbank_train = np.load('../data/fbank/train_delta2.npy')
#fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))
#train_lab = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(1,))
#train_lab_ids = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(0,))
#train_labels = phonemes.trans_ph48(train_lab)
#
#train_labels_sorted = np.zeros_like(train_labels)
#
#oldLabels = dict(zip(train_lab_ids,train_labels))
#for i in range(fbank_train_ids.shape[0]):
#    train_labels_sorted[i]=oldLabels[fbank_train_ids[i]]
#
#train_data = fbank_train[trainingSelection,:]
#train_ids = fbank_train_ids[trainingSelection]
#train_labels = train_labels_sorted[trainingSelection]
#
#val_data = fbank_train[validationSelection,:]
#val_ids = fbank_train_ids[validationSelection]
#val_labels = train_labels_sorted[validationSelection]
#
#test_data = np.load('../data/fbank/test_delta2.npy')
#test_ids = np.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))
#test_labels = np.zeros(shape=(test_data.shape[0],),dtype='int_')
#
#print 'FBANK Delta2: Loaded training, validation and test data.'

#%% Read MFCC Data

#mfcc_train = np.loadtxt(paths.pathToMFCCTrain, dtype='float32',delimiter=' ',usecols=range(1,40))
#mfcc_train_ids = np.loadtxt(paths.pathToMFCCTrain, dtype='str_', delimiter=' ', usecols=(0,))
#train_lab = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(1,))
#train_lab_ids = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(0,))
#train_labels = phonemes.trans_ph48(train_lab)
#
#train_labels_sorted = np.zeros_like(train_labels)
#
#oldLabels = dict(zip(train_lab_ids,train_labels))
#for i in range(mfcc_train_ids.shape[0]):
#    train_labels_sorted[i]=oldLabels[mfcc_train_ids[i]]
#
#train_data = mfcc_train[trainingSelection,:]
#train_ids = mfcc_train_ids[trainingSelection]
#train_labels = train_labels_sorted[trainingSelection]
#
#val_data = mfcc_train[validationSelection,:]
#val_ids = mfcc_train_ids[validationSelection]
#val_labels = train_labels_sorted[validationSelection]
#
#test_data = np.loadtxt(paths.pathToMFCCTest,dtype='float32',delimiter=' ',usecols=range(1,40))
#test_ids = np.loadtxt(paths.pathToMFCCTest,dtype='str_',delimiter=' ',usecols=(0,))
#test_labels = np.zeros(shape=(test_data.shape[0],),dtype='int_')

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

#phonemes2Id = speakersent.getPhonemeIds(train_labels)

# Attention: Sentence Segments start with 1
#def arrangeDataPhRep(data,ids,before,after,phonIds,labels):
#    arranged = np.zeros(shape=(data.shape[0],(before+after+1)*data.shape[1]),dtype='float32')
#    sswi = speakersent.getSpeakerSentenceWordId(ids)
#    w = data.shape[1]
#    for i in range(data.shape[0]):
#        sp,se,wo=speakersent.SpSeWo(ids[i])
#        for k in range(-before,after+1):
#            wo_=wo+k
#            if wo_ < 1:
#                wo_ = 1
#            elif wo_ > len(sswi[sp][se]):
#                wo_ = len(sswi[sp][se])
#            if np.random.rand(1)[0] < 0.4 and k != 0:
#                lab = labels[sswi[sp][se][wo_]]
#                tmpIdx=np.random.randint(0,len(phonIds[lab]))
#                replaceIdx =phonIds[lab][tmpIdx]
#                arranged[i,(k+before)*w:(k+1+before)*w] = data[replaceIdx,:]
#            else:
#                arranged[i,(k+before)*w:(k+1+before)*w] = data[sswi[sp][se][wo_],:]
#    return arranged
                    

    
#%% Clean the databases
    
trainDB = './caffedata/train.lvl'
valDB   = './caffedata/val.lvl'
testDB  = './caffedata/test.lvl'

import shutil

shutil.rmtree(trainDB)
shutil.rmtree(valDB)
shutil.rmtree(testDB)

#%% Write Data
rng = np.random.RandomState(0)
randomTrainingOrder = rng.permutation(np.arange(train_data_std.shape[0]))
write_lvldb.writeData(arrangeData(train_data_std,train_ids,NBefore,NAfter),train_labels,trainDB,randomTrainingOrder)
write_lvldb.writeData(arrangeData(val_data_std,val_ids,NBefore,NAfter),val_labels,valDB)
write_lvldb.writeData(arrangeData(test_data_std,test_ids,NBefore,NAfter),test_labels,testDB)
        
print 'Wrote data to databases.'
