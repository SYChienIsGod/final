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

NBefore = 3
NAfter = 3
# 0->FBANK, 1->MFCC, 2->IIF, 3->IIF2, 4->FBANKDelta2 (not active at the moment)
featureSelection = 2


def getDataSplit(trainingPath, testPath, dimensions, trainingSelection, validationSelection):
    train_features = np.loadtxt(trainingPath,dtype='float32',delimiter=' ',usecols=range(1,dimensions+1))
    train_feature_ids = np.loadtxt(trainingPath,dtype='str_',delimiter=' ',usecols=(0,))
    train_lab = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(1,))
    train_lab_ids = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(0,))
    train_labels = phonemes.trans_ph48(train_lab)
    train_labels_sorted = np.zeros_like(train_labels)

    if train_feature_ids.shape[0] > train_lab_ids.shape[0]:
        incl=np.in1d(train_feature_ids,train_lab_ids)
        train_feature_ids=train_feature_ids[incl]
        train_features=train_features[incl,:]

    oldLabels = dict(zip(train_lab_ids,train_labels))
    for i in range(train_feature_ids.shape[0]):
        train_labels_sorted[i]=oldLabels[train_feature_ids[i]]
    
    train_data = train_features[trainingSelection,:]
    train_ids = train_feature_ids[trainingSelection]
    train_labels = train_labels_sorted[trainingSelection]
    
    val_data = train_features[validationSelection,:]
    val_ids = train_feature_ids[validationSelection]
    val_labels = train_labels_sorted[validationSelection]
    
    test_data = np.loadtxt(testPath,dtype='float32',delimiter=' ',usecols=range(1,dimensions+1))
    test_ids = np.loadtxt(testPath,dtype='str_',delimiter=' ',usecols=(0,))
    test_labels = np.zeros(shape=(test_data.shape[0],),dtype='int_')
    
    return train_data,train_ids,train_labels,val_data,val_ids,val_labels,test_data,test_ids,test_labels

#%% Validation & Training data

# Select which groups should be in validation
fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))

speakerSets=speakersent.getSpeakerSets(fbank_train_ids)
sentenceGroupId,sentenceGroupInst=speakersent.getSpeakerSentenceGroups(fbank_train_ids,speakerSets)

validationSet = speakersent.getValidationSet(fbank_train_ids,sentenceGroupId,sentenceGroupInst)

validationSelection = np.in1d(sentenceGroupId,(validationSet))
trainingSelection = validationSelection==0

print 'Selected validation set: {}/{}'.format(np.count_nonzero(validationSelection),len(sentenceGroupId))

#%%

trainingPaths = (paths.pathToFBANKTrain,paths.pathToMFCCTrain,paths.toIIFTrain,paths.toIIF2Train)
testPaths = (paths.pathToFBANKTest,paths.pathToMFCCTest,paths.toIIFTest,paths.toIIF2Test)
dimensions = (69,39,55,60)

trainingPath = trainingPaths[featureSelection]
testPath=testPaths[featureSelection]
dimension=dimensions[featureSelection]

#%% Read FBANK Training Data

train_data,train_ids,train_labels,val_data,val_ids,val_labels,test_data,test_ids,test_labels = \
    getDataSplit(trainingPath,testPath, dimension, trainingSelection, validationSelection)
print 'Data loaded.'

#%%
#
#fbank_train = np.loadtxt(paths.pathToFBANKTrain,dtype='float32',delimiter=' ',usecols=range(1,70))
#fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))
#train_lab = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(1,))
#train_lab_ids = np.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',',usecols=(0,))
#
#train_labels = phonemes.trans_ph48(train_lab)
#
#train_labels_sorted = np.zeros_like(train_labels)
#
#oldLabels = dict(zip(train_lab_ids,train_labels))
#for i in range(fbank_train_ids.shape[0]):
#    train_labels_sorted[i]=oldLabels[fbank_train_ids[i]]
#
#train_data_2 = fbank_train[trainingSelection,:]
#train_ids_2 = fbank_train_ids[trainingSelection]
#train_labels_2 = train_labels_sorted[trainingSelection]
#
#val_data_2 = fbank_train[validationSelection,:]
#val_ids_2 = fbank_train_ids[validationSelection]
#val_labels_2 = train_labels_sorted[validationSelection]
#
#test_data_2 = np.loadtxt(paths.pathToFBANKTest,dtype='float32',delimiter=' ',usecols=range(1,70))
#test_ids_2 = np.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))
#test_labels_2 = np.zeros(shape=(test_data.shape[0],),dtype='int_')
#
#print 'FBANK: Loaded training, validation and test data.'

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

#shutil.rmtree(trainDB)
#shutil.rmtree(valDB)
#shutil.rmtree(testDB)

#%% Write Data
rng = np.random.RandomState(0)
randomTrainingOrder = rng.permutation(np.arange(train_data_std.shape[0]))
write_lvldb.writeData(arrangeData(train_data_std,train_ids,NBefore,NAfter),train_labels,trainDB,randomTrainingOrder)
write_lvldb.writeData(arrangeData(val_data_std,val_ids,NBefore,NAfter),val_labels,valDB)
write_lvldb.writeData(arrangeData(test_data_std,test_ids,NBefore,NAfter),test_labels,testDB)
print 'Wrote data to databases.'

#%%
with open('caffedata/train_ids.npy','w') as f:
    np.save(f,train_ids)
with open('caffedata/val_ids.npy','w') as f:
    np.save(f,val_ids)
with open('caffedata/test_ids.npy','w') as f:
    np.save(f,test_ids)

with open('caffedata/train_labels.npy','w') as f:
    np.save(f,train_labels)
with open('caffedata/val_labels.npy','w') as f:
    np.save(f,val_labels)
with open('caffedata/test_labels.npy','w') as f:
    np.save(f,test_labels)
    
with open('caffedata/training_order.npy','w') as f:
    np.save(f,randomTrainingOrder)