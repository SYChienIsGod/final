# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:51:11 2015

@author: jan
"""

import sys
import csv
import numpy as np
import time
caffe_root = '/home/jan/Dev/Repo/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import paths
import caffe_pb2
from google.protobuf import text_format
predLayerName = 'out' #Output of the last layer
import leveldb
# Load dictionary 
import util
import speakersent

# Load sequence name
#fbank_test_ids = np.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))
#fbank_train_ids = np.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))

netConfFile = './caffedata/basenet.prototxt'

trainDB = './caffedata/train.lvl'
valDB   = './caffedata/val.lvl'
testDB  = './caffedata/test.lvl'

def countEntries(dbfile):
    i = 0
    db = leveldb.LevelDB(dbfile)
    for k in db.RangeIter(include_value=False):
        i+=1
    return i
        


#%%
netConf = caffe_pb2.NetParameter()


with open (netConfFile, "r") as f:
    text_format.Merge(f.read(),netConf)

confFiles=list()
for source in ('train','val','test'):
    for i in range(len(netConf.layer)):
        if netConf.layer[i].name==u'data' and netConf.layer[i].include[0].phase==caffe.TEST:
            netConf.layer[i].data_param.source = './caffedata/'+source+'.lvl'
            netConf.layer[i].data_param.batch_size = 1
        if netConf.layer[i].name==u'data' and netConf.layer[i].include[0].phase==caffe.TRAIN:
            netConf.layer[i].data_param.source = './caffedata/'+source+'.lvl'
            netConf.layer[i].data_param.batch_size = 1
    fname = './caffedata/basenet_'+source+'.prototxt'
    with open(fname,'w') as f:
        f.write(text_format.MessageToString(netConf,as_utf8=True))

#%%

trainingPermutation = util.loadNpy('caffedata/training_order.npy')

for source in ('val','train','test'):
    N=countEntries('./caffedata/'+source+'.lvl')
    caffe.set_mode_gpu()
    net=caffe.Net('./caffedata/basenet_'+source+'.prototxt', './caffedata/snapshot_iter_415800.caffemodel',caffe.TEST)
    features=np.zeros(shape=(N,48),dtype='float32')
    util.startprogress('Computing '+source)
    for i in range(N):
        net.forward()
        features[i,:]=net.blobs['ip4'].data
        if i%1000==0:
            util.progress(float(i)/float(N)/2.0*100.0)
    if source=='train':
        aso=np.argsort(trainingPermutation)
        features=features[aso,:]
    labels = util.loadNpy('caffedata/'+source+'_labels.npy')
    ids = util.loadNpy('caffedata/'+source+'_ids.npy')
    sswi = speakersent.getSpeakerSentenceWordId(ids)
    n_sent = speakersent.countSentences(sswi)
    f= file('ssvmdata/'+source+'.txt','w')
    sp_=''
    se_=''
    sentId=-1
    for i  in range(ids.shape[0]):  
        if i%1000==0:
            util.progress(float(i)/float(N)/2.0*100.0+50.0)
        id_=ids[i]
        sp,se,wo=speakersent.SpSeWo(id_)
        if sp_!=sp or se!=se_:
            sentId+=1
            sp_=sp
            se_=se
        n_frames = len(sswi[sp][se])
        f.write('%i %i %i %i %i ' % (n_sent, sentId, n_frames, wo-1, labels[i]) + ' '.join([str(x_) for x_ in features[i,:]]) + '\n')        
    f.close()
    util.endprogress()
